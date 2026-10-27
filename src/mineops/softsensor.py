"""Soft sensor for silica in concentrate, tested the way it would be used in the plant.

Set-up
* One row per hour. The target is the laboratory silica result, on measured hours only (see plant_quality).
* The laboratory result for hour t reaches the control room `delay` hours later. The soft sensor estimates the
  current hour from the process sensors and from laboratory results that have already arrived.
* Iron in concentrate is never an input: it comes from the same laboratory sample as silica (correlation -0.80),
  so using it would be leakage.
* Models are retrained each month on all earlier months and scored on the next month (rolling origin). The month
  before each test month is used to set the shrinkage weight and to calibrate the 80 percent interval
  (split conformal), so nothing is tuned on the month being scored.

Models
* persistence: the latest laboratory result that has arrived (what the control room uses today);
* training mean; ridge regression and gradient boosting on the process sensors only;
* gradient boosting on laboratory history alone, and on process sensors plus laboratory history, both predicting
  the change from the latest result;
* shrunk: persistence plus a share w of the predicted change, with w chosen on the previous month. This is done
  for both boosting models, so "shrunk" (the soft sensor) against "shrunk_lab_only" shows what the process
  sensors add, like for like.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import AIR, FEED, LEVELS, PROCESS, PULP, REAGENTS
from .plant_kpis import OFF_SPEC

# starting values; run_plant.py replaces the four tuned settings with the best row of tune() (May only)
GB_PARAMS = dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=80,
                 l2_regularization=1.0, random_state=0)
MONTHS = ["2017-05", "2017-06", "2017-07", "2017-08", "2017-09"]   # the first is a calibration month only
LAB_FEATURES = ["lab_last", "lab_mean6", "lab_age", "lab_slope"]
COVERAGE = 0.80


def process_features(h: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean and spread of each sensor, its change from the previous hour, its three-hour mean, and the feed assays."""
    X = pd.DataFrame(index=h.index)
    for t in PROCESS:
        m = h[f"{t}_mean"]
        X[f"{t}_mean"] = m
        X[f"{t}_std"] = h[f"{t}_std"]
        X[f"{t}_d1"] = m - m.shift(1)
        X[f"{t}_r3"] = m.rolling(3, min_periods=1).mean()
    for t in FEED:
        X[t] = h[t]
    return X


def lab_features(h: pd.DataFrame, lab: pd.Series, delay: int) -> pd.DataFrame:
    """Laboratory history available at hour t when results arrive `delay` hours after their sample."""
    s = h["silica_conc"].where(lab == "measured")
    avail = s.shift(delay)
    last = avail.ffill(limit=24)
    pos = pd.Series(np.where(avail.notna(), np.arange(len(avail)), np.nan), index=h.index).ffill()
    age = (np.arange(len(avail)) - pos) + delay
    known = avail.dropna()
    prev = known.shift(1).reindex(h.index).ffill()
    return pd.DataFrame({
        "lab_last": last,
        "lab_mean6": avail.rolling(6, min_periods=1).mean().fillna(last),
        "lab_age": age.where(last.notna()),
        "lab_slope": (last - prev).where(last.notna()),
    }, index=h.index)


def feature_groups(columns) -> dict[str, list[str]]:
    def pick(tags):
        return [c for c in columns if any(c == t or c.startswith(t + "_") for t in tags)]
    return {
        "Laboratory history": [c for c in columns if c.startswith("lab_")],
        "Feed assays": pick(FEED),
        "Reagents (starch, amine)": pick(REAGENTS),
        "Ore pulp (flow, pH, density)": pick(PULP),
        "Column air flows": pick(AIR),
        "Column levels": pick(LEVELS),
    }


def _conformal_q(abs_resid: np.ndarray, coverage: float = COVERAGE) -> float:
    n = len(abs_resid)
    if n == 0:
        return float("nan")
    k = min(n, math.ceil((n + 1) * coverage))
    return float(np.sort(abs_resid)[k - 1])


def _best_weight(y, last, delta) -> float:
    grid = np.round(np.arange(0, 1.01, 0.1), 2)
    errs = [mean_absolute_error(y, last + w * delta) for w in grid]
    return float(grid[int(np.argmin(errs))])


def _best_threshold(y, score) -> float:
    """Warning threshold that maximizes F1 on the calibration month."""
    best, best_f1 = 0.5, -1.0
    for th in np.unique(np.quantile(score, np.linspace(0.5, 0.98, 49))):
        warn = score >= th
        tp = np.sum(warn & (y == 1))
        if warn.sum() == 0 or y.sum() == 0:
            continue
        p, r = tp / warn.sum(), tp / y.sum()
        f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
        if f1 > best_f1:
            best, best_f1 = float(th), f1
    return best


def dataset(h: pd.DataFrame, lab: pd.Series, delay: int, frozen: pd.Series | None = None):
    X = process_features(h).join(lab_features(h, lab, delay))
    y = h["silica_conc"]
    ok = (lab == "measured") & y.notna() & X["lab_last"].notna()
    if frozen is not None:
        ok &= ~frozen.reindex(h.index, fill_value=False)
    return X, y, ok


def rolling_eval(h, lab, delay: int, frozen=None, months=MONTHS, params=None, classify=False, keep_models=False):
    """Monthly rolling-origin evaluation. Returns hourly predictions, monthly scores and (optionally) models."""
    params = params or GB_PARAMS
    X, y, ok = dataset(h, lab, delay, frozen)
    proc_cols = [c for c in X.columns if not c.startswith("lab_")]
    all_cols = list(X.columns)
    month_of = pd.Series(X.index.strftime("%Y-%m"), index=X.index)
    frames, models = [], {}
    for m in months:
        start = pd.Timestamp(m + "-01")
        tr = ok & (X.index < start)
        te = ok & (month_of == m)
        if tr.sum() < 100 or te.sum() == 0:
            continue
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        med = Xtr[proc_cols].median()
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr[proc_cols].fillna(med), ytr)
        gbp = HistGradientBoostingRegressor(**params).fit(Xtr[proc_cols], ytr)
        gbl = HistGradientBoostingRegressor(**params).fit(Xtr[all_cols], ytr - Xtr["lab_last"])
        delta = gbl.predict(Xte[all_cols])
        gbo = HistGradientBoostingRegressor(**params).fit(Xtr[LAB_FEATURES], ytr - Xtr["lab_last"])
        f = pd.DataFrame({
            "month": m, "y": yte, "persistence": Xte["lab_last"], "train_mean": ytr.mean(),
            "ridge_process": ridge.predict(Xte[proc_cols].fillna(med)), "gb_process": gbp.predict(Xte[proc_cols]),
            "delta": delta, "gb_lab": Xte["lab_last"] + delta,
            "delta_lab_only": gbo.predict(Xte[LAB_FEATURES]),
        }, index=Xte.index)
        f["gb_lab_only"] = f["persistence"] + f["delta_lab_only"]
        if classify:
            off_tr = (ytr > OFF_SPEC).astype(int)
            c_all = HistGradientBoostingClassifier(**params).fit(Xtr[all_cols], off_tr)
            c_proc = HistGradientBoostingClassifier(**params).fit(Xtr[proc_cols], off_tr)
            f["p_offspec"] = c_all.predict_proba(Xte[all_cols])[:, 1]
            f["p_offspec_process"] = c_proc.predict_proba(Xte[proc_cols])[:, 1]
            f["base_rate"] = off_tr.mean()
        frames.append(f)
        if keep_models:
            models[m] = {"gb_lab": gbl, "columns": all_cols}
    P = pd.concat(frames)
    # shrinkage weight, interval and warning thresholds come from the previous month
    out = []
    for i, m in enumerate(months[1:], start=1):
        cur, prev = P[P["month"] == m].copy(), P[P["month"] == months[i - 1]]
        if cur.empty or prev.empty:
            continue
        w = _best_weight(prev["y"], prev["persistence"], prev["delta"])
        cur["weight"] = w
        cur["shrunk"] = cur["persistence"] + w * cur["delta"]
        wo = _best_weight(prev["y"], prev["persistence"], prev["delta_lab_only"])
        cur["weight_lab_only"] = wo
        cur["shrunk_lab_only"] = cur["persistence"] + wo * cur["delta_lab_only"]
        q = _conformal_q(np.abs(prev["y"] - (prev["persistence"] + w * prev["delta"])).to_numpy())
        cur["lo"], cur["hi"] = (cur["shrunk"] - q).clip(lower=0), cur["shrunk"] + q
        if classify:
            cur["th_model"] = _best_threshold((prev["y"] > OFF_SPEC).astype(int).to_numpy(), prev["p_offspec"].to_numpy())
            cur["th_persist"] = _best_threshold((prev["y"] > OFF_SPEC).astype(int).to_numpy(), prev["persistence"].to_numpy())
        out.append(cur)
    preds = pd.concat(out)
    return preds, models


POINT_MODELS = ["persistence", "train_mean", "ridge_process", "gb_process", "gb_lab_only", "gb_lab", "shrunk_lab_only", "shrunk"]


def scores(preds: pd.DataFrame, by_month: bool = True) -> pd.DataFrame:
    def one(d):
        row = {"hours": len(d)}
        for k in POINT_MODELS:
            e = d[k] - d["y"]
            row[f"{k}_mae"] = float(np.mean(np.abs(e)))
            row[f"{k}_rmse"] = float(np.sqrt(np.mean(e ** 2)))
            row[f"{k}_r2"] = float(r2_score(d["y"], d[k]))
            row[f"{k}_bias"] = float(np.mean(e))
        row["coverage_80"] = float(np.mean((d["y"] >= d["lo"]) & (d["y"] <= d["hi"])))
        row["interval_width"] = float(np.mean(d["hi"] - d["lo"]))
        row["weight"] = float(d["weight"].mean())
        return row
    rows = {"all": one(preds)}
    if by_month:
        for m, d in preds.groupby("month"):
            rows[m] = one(d)
    return pd.DataFrame(rows).T


def warning_scores(preds: pd.DataFrame) -> dict:
    y = (preds["y"] > OFF_SPEC).astype(int).to_numpy()
    out = {"off_spec_rate_pct": 100 * y.mean(), "hours": int(len(y))}
    for name, score, th in [("model", preds["p_offspec"], preds["th_model"]),
                            ("persistence", preds["persistence"], preds["th_persist"])]:
        warn = (score >= th).to_numpy()
        tp = int(np.sum(warn & (y == 1)))
        out[name] = {
            "auc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
            "catch_rate_pct": 100 * tp / max(1, y.sum()),
            "false_alarm_pct": 100 * (1 - tp / max(1, warn.sum())),
            "warnings": int(warn.sum()),
        }
    out["process_only"] = {"auc": float(roc_auc_score(y, preds["p_offspec_process"])),
                           "average_precision": float(average_precision_score(y, preds["p_offspec_process"]))}
    p = preds["p_offspec"].to_numpy()
    base = preds["base_rate"].to_numpy()
    out["model"]["brier"] = float(np.mean((p - y) ** 2))
    out["model"]["brier_skill_pct"] = float(100 * (1 - np.mean((p - y) ** 2) / np.mean((base - y) ** 2)))
    return out


def delay_sweep(h, lab, frozen, delays=(1, 2, 3, 4, 6, 8, 12), params=None) -> pd.DataFrame:
    rows = []
    for d in delays:
        p, _ = rolling_eval(h, lab, d, frozen, params=params)
        s = scores(p, by_month=False).loc["all"]
        rows.append({"delay_hours": d, "hours": int(s["hours"]), "persistence_mae": s["persistence_mae"],
                     "gb_lab_mae": s["gb_lab_mae"], "gb_lab_only_mae": s["gb_lab_only_mae"],
                     "shrunk_lab_only_mae": s["shrunk_lab_only_mae"], "shrunk_mae": s["shrunk_mae"], "train_mean_mae": s["train_mean_mae"],
                     "gain_vs_persistence_pct": 100 * (1 - s["shrunk_mae"] / s["persistence_mae"]),
                     "sensors_add_pct": 100 * (1 - s["shrunk_mae"] / s["shrunk_lab_only_mae"]),
                     "coverage_80": s["coverage_80"]})
    return pd.DataFrame(rows).set_index("delay_hours")


def grouped_importance(h, lab, delay, frozen, models, preds, repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Increase in mean absolute error of the shrunk estimate when one group of the model's inputs is shuffled within
    each test month. The latest result that the estimate is built on is kept fixed, so the score measures what the
    model learned, not the loss of the latest result itself."""
    X, y, ok = dataset(h, lab, delay, frozen)
    rng = np.random.default_rng(seed)
    groups = feature_groups([c for c in X.columns])
    rows = {g: [] for g in groups}
    for m, d in preds.groupby("month"):
        mdl, cols = models[m]["gb_lab"], models[m]["columns"]
        Xm = X.loc[d.index, cols]
        w = d["weight"].iloc[0]
        anchor = Xm["lab_last"].to_numpy()
        base = mean_absolute_error(d["y"], anchor + w * mdl.predict(Xm))
        for g, gc in groups.items():
            for _ in range(repeats):
                Xp = Xm.copy()
                perm = rng.permutation(len(Xp))
                Xp[gc] = Xp[gc].to_numpy()[perm]
                rows[g].append(mean_absolute_error(d["y"], anchor + w * mdl.predict(Xp)) - base)
    return pd.DataFrame({"mae_increase": {g: float(np.mean(v)) for g, v in rows.items()},
                         "spread": {g: float(np.std(v)) for g, v in rows.items()}}).sort_values("mae_increase", ascending=False)


def tune(h, lab, delay, frozen, train_end="2017-05-01", valid_month="2017-05") -> pd.DataFrame:
    """Small grid on the calibration month only (May). The test months (June to September) play no part."""
    X, y, ok = dataset(h, lab, delay, frozen)
    tr = ok & (X.index < pd.Timestamp(train_end))
    va = ok & (X.index.strftime("%Y-%m") == valid_month)
    rows = []
    for lr in (0.03, 0.06):
        for leaves in (7, 15, 31):
            for msl in (20, 40, 80):
                for l2 in (0.0, 1.0):
                    p = dict(GB_PARAMS, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, l2_regularization=l2)
                    mdl = HistGradientBoostingRegressor(**p).fit(X[tr], y[tr] - X.loc[tr, "lab_last"])
                    pred = X.loc[va, "lab_last"] + mdl.predict(X[va])
                    rows.append({"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl,
                                 "l2_regularization": l2, "valid_mae": mean_absolute_error(y[va], pred)})
    return pd.DataFrame(rows).sort_values("valid_mae").reset_index(drop=True)


def learning_curve(h, lab, delay, frozen, params=None, test_start="2017-08-01", weeks=(2, 4, 6, 8, 10, 12, 14, 16, 18)):
    params = params or GB_PARAMS
    X, y, ok = dataset(h, lab, delay, frozen)
    t0 = pd.Timestamp(test_start)
    te = ok & (X.index >= t0)
    rows = []
    for w in weeks:
        tr = ok & (X.index < t0) & (X.index >= t0 - pd.Timedelta(weeks=w))
        if tr.sum() < 100:
            continue
        mdl = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr] - X.loc[tr, "lab_last"])
        rows.append({"weeks": w, "train_hours": int(tr.sum()),
                     "train_mae": mean_absolute_error(y[tr], X.loc[tr, "lab_last"] + mdl.predict(X[tr])),
                     "test_mae": mean_absolute_error(y[te], X.loc[te, "lab_last"] + mdl.predict(X[te])),
                     "persistence_mae": mean_absolute_error(y[te], X.loc[te, "lab_last"])})
    return pd.DataFrame(rows).set_index("weeks")


def training_curve(h, lab, delay, frozen, params=None, split="2017-07-01", valid_end="2017-08-01", iters=600):
    params = dict(params or GB_PARAMS, max_iter=iters, early_stopping=False)
    X, y, ok = dataset(h, lab, delay, frozen)
    tr = ok & (X.index < pd.Timestamp(split))
    va = ok & (X.index >= pd.Timestamp(split)) & (X.index < pd.Timestamp(valid_end))
    mdl = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr] - X.loc[tr, "lab_last"])
    rows = []
    for i, (ptr, pva) in enumerate(zip(mdl.staged_predict(X[tr]), mdl.staged_predict(X[va])), start=1):
        if i % 10 == 0 or i == 1:
            rows.append({"iteration": i,
                         "train_mae": mean_absolute_error(y[tr], X.loc[tr, "lab_last"] + ptr),
                         "valid_mae": mean_absolute_error(y[va], X.loc[va, "lab_last"] + pva)})
    return pd.DataFrame(rows).set_index("iteration")


def leakage_demo(df: pd.DataFrame, h, lab, rolling_scores: pd.Series, frozen=None, params=None, delay: int = 2,
                 sample: int = 200_000, seed: int = 0) -> pd.DataFrame:
    """How the usual way of testing on this dataset overstates accuracy, step by step."""
    rows = []
    raw = df.dropna().sample(min(sample, len(df)), random_state=seed)
    inputs_all = [c for c in raw.columns if c not in ("time", "silica_conc")]
    for label, cols in [("Random split of 20-second rows, all columns (iron in concentrate included)", inputs_all),
                        ("Random split of 20-second rows, iron in concentrate removed", [c for c in inputs_all if c != "iron_conc"])]:
        Xtr, Xte, ytr, yte = train_test_split(raw[cols], raw["silica_conc"], test_size=0.2, random_state=seed)
        mdl = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(Xtr, ytr)
        p = mdl.predict(Xte)
        rows.append({"test": label, "r2": r2_score(yte, p), "mae": mean_absolute_error(yte, p)})
    X = process_features(h)
    ok = (lab == "measured") & h["silica_conc"].notna()
    Xtr, Xte, ytr, yte = train_test_split(X[ok], h.loc[ok, "silica_conc"], test_size=0.2, random_state=seed)
    mdl = HistGradientBoostingRegressor(**GB_PARAMS).fit(Xtr, ytr)
    p = mdl.predict(Xte)
    rows.append({"test": "Random split of measured hours, process sensors only", "r2": r2_score(yte, p), "mae": mean_absolute_error(yte, p)})
    s = rolling_scores
    rows.append({"test": "Next month, measured hours, process sensors only", "r2": s["gb_process_r2"], "mae": s["gb_process_mae"]})
    rows.append({"test": "Next month, latest laboratory result (persistence)", "r2": s["persistence_r2"], "mae": s["persistence_mae"]})
    rows.append({"test": "Next month, soft sensor (process sensors plus laboratory history)", "r2": s["shrunk_r2"], "mae": s["shrunk_mae"]})
    # the same soft sensor, retrained and scored with every hour that has a value treated as a measurement
    filled = pd.Series(np.where(lab == "missing", "missing", "measured"), index=lab.index)
    pf, _ = rolling_eval(h, filled, delay, frozen, params=params)
    sf = scores(pf, by_month=False).loc["all"]
    rows.append({"test": "Next month, soft sensor, every hour with a value treated as measured (for training, history and scoring)",
                 "r2": sf["shrunk_r2"], "mae": sf["shrunk_mae"]})
    return pd.DataFrame(rows)
