"""Can the plant data be trusted? Sampling, gaps, stuck sensors, range checks and, above all, which laboratory
values are real measurements and which were filled in.

The silica assay is a laboratory result for an hourly sample, but the file repeats it on every 20-second row.
Some hours repeat the previous hour's result, some lie on a straight line between two results (interpolated
across a gap), and some change smoothly within the hour (interpolated at 20-second steps). Only hours that carry
a new, constant value are treated as measurements. Training or scoring a model on filled-in values rewards it
for reproducing the filling, not for predicting the plant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEED, LAB, PROCESS

ROWS_PER_HOUR = 180          # 20-second readings
LAB_CLASSES = ["measured", "repeat", "interpolated", "varies within hour", "missing"]
RANGES = {                   # physically possible limits used for range checks
    "pulp_ph": (0.0, 14.0), "pulp_density": (1.0, 3.0),
    "iron_feed": (0.0, 100.0), "silica_feed": (0.0, 100.0), "iron_conc": (0.0, 100.0), "silica_conc": (0.0, 100.0),
}


def hourly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per hour on a complete hourly index (missing hours present as NaN)."""
    g = df.groupby("time")
    h = g[PROCESS].mean().add_suffix("_mean")
    h = h.join(g[PROCESS].std(ddof=0).add_suffix("_std"))
    h = h.join(g[FEED].mean())
    h = h.join(g[LAB].mean())
    h["silica_nunique"] = g["silica_conc"].nunique()
    h["rows"] = g.size()
    h["dup_rows"] = df.duplicated().groupby(df["time"]).sum()
    h = h.asfreq("h")
    h["rows"] = h["rows"].fillna(0).astype(int)
    h["dup_rows"] = h["dup_rows"].fillna(0).astype(int)
    return h


def classify_lab(h: pd.DataFrame, col: str = "silica_conc", tol: float = 1e-6) -> pd.Series:
    """Label each hour's laboratory value: measured, repeat, interpolated, varies within hour or missing."""
    s = h[col]
    within = h["silica_nunique"].fillna(0) > 1
    repeat = s.diff().abs() < 1e-9
    second = (s.shift(-1) - 2 * s + s.shift(1)).abs()
    linear = (second < tol) & ~repeat
    lab = pd.Series("measured", index=h.index)
    lab[linear] = "interpolated"
    lab[repeat] = "repeat"
    lab[within] = "varies within hour"
    lab[s.isna()] = "missing"
    return lab


def gaps(h: pd.DataFrame) -> list[dict]:
    """Runs of hours with no readings at all."""
    missing = h["rows"] == 0
    out, start = [], None
    for t, m in missing.items():
        if m and start is None:
            start = t
        if not m and start is not None:
            out.append({"start": start, "end": t - pd.Timedelta(hours=1), "hours": int((t - start) / pd.Timedelta(hours=1))})
            start = None
    if start is not None:
        out.append({"start": start, "end": h.index[-1], "hours": int((h.index[-1] - start) / pd.Timedelta(hours=1)) + 1})
    return out


def update_interval_hours(h: pd.DataFrame, col: str) -> float:
    """Median number of hours between changes in a value (the feed assays change far less often than hourly)."""
    s = h[col].dropna()
    changes = s.index[s.diff().abs() > 1e-9]
    if len(changes) < 2:
        return float("nan")
    return float(np.median(np.diff(changes) / pd.Timedelta(hours=1)))


def longest_flat_run(values: pd.Series) -> int:
    """Longest run of identical consecutive readings."""
    v = values.to_numpy()
    if len(v) == 0:
        return 0
    change = np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(change)
    return int(np.bincount(run_id).max())


def flatlines(df: pd.DataFrame, tags: list[str] = PROCESS) -> pd.DataFrame:
    """Stuck-sensor screen on the 20-second readings: longest unchanged run and the share of time in runs of an hour or more."""
    rows = []
    for t in tags:
        v = df[t].to_numpy()
        change = np.r_[True, v[1:] != v[:-1]]
        run_id = np.cumsum(change)
        lengths = np.bincount(run_id)[run_id]
        rows.append({
            "tag": t,
            "longest_run_hours": round(longest_flat_run(df[t]) / ROWS_PER_HOUR, 1),
            "pct_rows_in_runs_over_1h": round(100 * float(np.mean(lengths >= ROWS_PER_HOUR)), 2),
        })
    return pd.DataFrame(rows).set_index("tag")


def stuck_periods(df: pd.DataFrame, tags: list[str] = PROCESS, min_hours: float = 24.0) -> list[dict]:
    """Runs where a sensor repeats exactly the same reading for at least min_hours (a frozen instrument or historian tag)."""
    out = []
    for t in tags:
        v = df[t].to_numpy()
        change = np.r_[True, v[1:] != v[:-1]]
        run_id = np.cumsum(change)
        sizes = np.bincount(run_id)
        for rid in np.where(sizes >= min_hours * ROWS_PER_HOUR)[0]:
            idx = np.where(run_id == rid)[0]
            out.append({"tag": t, "start": df["time"].iloc[idx[0]], "end": df["time"].iloc[idx[-1]],
                        "hours_of_data": round(len(idx) / ROWS_PER_HOUR, 1), "value": float(v[idx[0]])})
    return out


def frozen_hours(h: pd.DataFrame, min_tags: int = 15, dup_share: float = 0.25) -> pd.Series:
    """Hours when the data feed stalled and values were held: most sensors did not change at all, or at least a
    quarter of the rows are exact copies of an earlier row."""
    flat = (h[[f"{t}_std" for t in PROCESS]] == 0).sum(axis=1)
    dup = h["dup_rows"] if "dup_rows" in h else pd.Series(0, index=h.index)
    return (flat >= min_tags) | (dup >= dup_share * ROWS_PER_HOUR)


def clean_hourly(h: pd.DataFrame, stuck: list[dict], frozen: pd.Series) -> pd.DataFrame:
    """Blank the hourly values of stuck sensors and of hours when the whole feed was frozen."""
    h = h.copy()
    for p in stuck:
        span = (h.index >= p["start"].floor("h")) & (h.index <= p["end"].floor("h"))
        h.loc[span, [f"{p['tag']}_mean", f"{p['tag']}_std"]] = np.nan
    cols = [f"{t}_mean" for t in PROCESS] + [f"{t}_std" for t in PROCESS]
    h.loc[frozen, cols] = np.nan
    return h


def range_violations(df: pd.DataFrame) -> dict:
    out = {}
    for t in PROCESS + FEED + LAB:
        lo, hi = RANGES.get(t, (0.0, np.inf))
        out[t] = int(((df[t] < lo) | (df[t] > hi)).sum())
    return out


def audit(df: pd.DataFrame, h: pd.DataFrame, lab: pd.Series) -> dict:
    """Everything the data quality section of the report needs, as plain numbers."""
    present = h[h["rows"] > 0]
    counts = lab.value_counts().reindex(LAB_CLASSES, fill_value=0)
    measured = lab == "measured"
    gap_list = gaps(h)
    flat = flatlines(df)
    stuck = stuck_periods(df)
    frozen = frozen_hours(h)
    return {
        "rows": int(len(df)),
        "period": [str(h.index.min()), str(h.index.max())],
        "hours_in_period": int(len(h)),
        "hours_with_data": int(len(present)),
        "hours_complete": int((present["rows"] == ROWS_PER_HOUR).sum()),
        "incomplete_hours": {str(t): int(r) for t, r in present.loc[present["rows"] != ROWS_PER_HOUR, "rows"].items()},
        "gaps": [{"start": str(g["start"]), "end": str(g["end"]), "hours": g["hours"]} for g in gap_list],
        "duplicate_rows": int(df.duplicated().sum()),
        "lab_classes": {k: int(v) for k, v in counts.items()},
        "measured_share_pct": round(100 * measured.sum() / max(1, (lab != "missing").sum()), 1),
        "feed_assay_update_hours": {c: update_interval_hours(h, c) for c in FEED},
        "iron_silica_correlation": round(float(h["iron_conc"].corr(h["silica_conc"])), 3),
        "range_violations": range_violations(df),
        "flatlines": flat.reset_index().to_dict(orient="records"),
        "stuck_periods": [{**p, "start": str(p["start"]), "end": str(p["end"])} for p in stuck],
        "frozen_hours": [str(t) for t in frozen[frozen].index],
    }
