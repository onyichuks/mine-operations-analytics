"""Plant key performance indicators from the trusted hours only.

Off-specification is defined here as silica in concentrate above 3 percent. That limit is an assumption for
illustration (about the 81st percentile of measured hours); each plant sets its own limit by contract.
Recovery cannot be calculated because the dataset has no tailings assay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABELS, PROCESS

OFF_SPEC = 3.0


def add_ratios(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["starch_per_pulp"] = h["starch_flow_mean"] / h["pulp_flow_mean"]
    h["amina_per_pulp"] = h["amina_flow_mean"] / h["pulp_flow_mean"]
    h["iron_upgrade"] = h["iron_conc"] / h["iron_feed"]
    return h


def monthly(h: pd.DataFrame, lab: pd.Series) -> pd.DataFrame:
    """Monthly KPIs. Concentrate quality uses measured laboratory hours only."""
    h = add_ratios(h)
    m = lab == "measured"
    q = h.loc[m]
    month = q.index.to_period("M")
    out = pd.DataFrame({
        "measured_hours": q.groupby(month).size(),
        "silica_mean": q.groupby(month)["silica_conc"].mean(),
        "silica_p90": q.groupby(month)["silica_conc"].quantile(0.9),
        "off_spec_pct": q.groupby(month)["silica_conc"].apply(lambda s: 100 * (s > OFF_SPEC).mean()),
        "iron_conc_mean": q.groupby(month)["iron_conc"].mean(),
    })
    allm = h.index.to_period("M")
    run = h[h["rows"] > 0]
    rm = run.index.to_period("M")
    out["hours_running"] = run.groupby(rm).size()
    out["iron_feed_mean"] = run.groupby(rm)["iron_feed"].mean()
    out["silica_feed_mean"] = run.groupby(rm)["silica_feed"].mean()
    out["pulp_flow_mean"] = run.groupby(rm)["pulp_flow_mean"].mean()
    out["starch_per_pulp"] = run.groupby(rm)["starch_per_pulp"].mean()
    out["amina_per_pulp"] = run.groupby(rm)["amina_per_pulp"].mean()
    out["hours_in_month"] = h.groupby(allm).size()
    out.index = out.index.astype(str)
    return out


def daily(h: pd.DataFrame, lab: pd.Series) -> pd.DataFrame:
    h = add_ratios(h)
    m = lab == "measured"
    d = pd.DataFrame(index=pd.date_range(h.index.min().normalize(), h.index.max().normalize(), freq="D"))
    q = h.loc[m]
    d["silica_mean"] = q["silica_conc"].resample("D").mean()
    d["off_spec_pct"] = (q["silica_conc"] > OFF_SPEC).resample("D").mean() * 100
    d["measured_hours"] = q["silica_conc"].resample("D").count()
    d["silica_feed"] = h["silica_feed"].resample("D").mean()
    d["amina_per_pulp"] = h["amina_per_pulp"].resample("D").mean()
    d["starch_per_pulp"] = h["starch_per_pulp"].resample("D").mean()
    d["pulp_flow"] = h["pulp_flow_mean"].resample("D").mean()
    return d


def off_spec_conditions(h: pd.DataFrame, lab: pd.Series) -> pd.DataFrame:
    """How process conditions differ in off-specification hours: standardized mean difference (Cohen's d).
    This describes association only; operators also react to high silica, so cause and effect run both ways."""
    m = (lab == "measured") & h["silica_conc"].notna()
    q = h.loc[m]
    off = q["silica_conc"] > OFF_SPEC
    rows = []
    for t in ["silica_feed", "iron_feed"] + PROCESS:
        col = t if t in ("silica_feed", "iron_feed") else f"{t}_mean"
        a, b = q.loc[off, col].dropna(), q.loc[~off, col].dropna()
        pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / max(1, len(a) + len(b) - 2))
        rows.append({"tag": t, "label": LABELS[t], "off_spec_mean": a.mean(), "on_spec_mean": b.mean(),
                     "std_difference": (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan})
    out = pd.DataFrame(rows).set_index("tag")
    return out.reindex(out["std_difference"].abs().sort_values(ascending=False).index)


def lagged_response(h: pd.DataFrame, lab: pd.Series, tags=("amina_flow", "starch_flow", "air_flow_1", "level_5"),
                    lags=range(-3, 9)) -> pd.DataFrame:
    """Correlation between a process setting at hour t and the measured silica at hour t - lag.
    A peak at a positive lag means the setting follows the laboratory result: operators reacting to it."""
    s = h["silica_conc"].where(lab == "measured")
    out = {}
    for t in tags:
        x = h[f"{t}_mean"]
        out[t] = [float(x.corr(s.shift(L))) for L in lags]
    return pd.DataFrame(out, index=list(lags)).rename_axis("lag_hours")


def summary(h: pd.DataFrame, lab: pd.Series) -> dict:
    m = (lab == "measured") & h["silica_conc"].notna()
    s = h.loc[m, "silica_conc"]
    return {
        "off_spec_limit_pct": OFF_SPEC,
        "measured_hours": int(m.sum()),
        "silica_mean": float(s.mean()),
        "silica_median": float(s.median()),
        "silica_p75": float(s.quantile(0.75)),
        "silica_p90": float(s.quantile(0.90)),
        "silica_max": float(s.max()),
        "off_spec_pct": float(100 * (s > OFF_SPEC).mean()),
        "iron_conc_mean": float(h.loc[m, "iron_conc"].mean()),
        "iron_feed_mean": float(h["iron_feed"].mean()),
        "silica_feed_mean": float(h["silica_feed"].mean()),
    }
