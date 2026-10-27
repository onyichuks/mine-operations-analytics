"""Which gold mines will have the highest lost-time injury rate next year?

Each row is a mine and a year t. Inputs describe year t and earlier: size and workforce mix, injury history, and
MSHA inspection and violation records. The target is the number of NFDL injuries to the mine's own employees in
year t + 1, with that year's employee hours as the exposure, so the models predict an injury rate. Only mines
with at least `min_hours` employee hours in year t and some hours in year t + 1 are included.

Injury rates fell steeply over the period, so every model works in relative terms: a mine's rate is expressed
against the industry rate of the same year, and predictions are converted back with the latest industry rate
known when the prediction is made (year t). The baseline gives every mine that latest industry rate, and models
are scored with the Poisson deviance against it, so the skill measures how well a model separates high-rate from
low-rate mines, not how well it knows mine size or the industry trend. The test is temporal: for each target
year from `test_from`, models are trained on all earlier target years.

The question is whether inspection and violation records add anything to what a mine's own injury history
already tells you. This is a screening tool for where to look first, not a causal estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compliance import mine_year
from .safety import BASE_HOURS, flag_injuries

EXPOSURE = ["log_hours", "log_employees", "hours_growth", "share_underground", "share_mill", "share_pit", "years_reporting"]
HISTORY = ["nfdl", "all_injury", "medical_only", "nfdl_prev", "nfdl_3yr", "hours_3yr", "relative_rate", "relative_3yr",
           "contractor_injuries", "new_worker_injuries", "days_lost"]
COMPLIANCE = ["violations", "ss_violations", "orders", "high_negligence", "log_penalties", "log_inspection_hours",
              "inspections", "complaint_inspections", "violations_per_100h", "ss_per_100h", "ss_share"]
GB_PARAMS = dict(loss="poisson", max_iter=200, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=40,
                 l2_regularization=1.0, random_state=0)
PRIOR_INJURIES = 1.0   # weight of the industry rate in the shrunk own-rate baseline, in expected injuries


def industry_rates(prod: pd.DataFrame, acc: pd.DataFrame) -> pd.Series:
    """Operator lost-time injuries per 200,000 hours across all gold mines, by year."""
    a = flag_injuries(acc)
    n = a[~a["contractor"] & a["nfdl"]].groupby("year").size()
    h = prod.groupby("year")["hours"].sum()
    return (n.reindex(h.index, fill_value=0) / h * BASE_HOURS).astype(float)


def panel(prod: pd.DataFrame, acc: pd.DataFrame, vio: pd.DataFrame, ins: pd.DataFrame,
          min_hours: float = 10_000, last_year: int | None = None) -> pd.DataFrame:
    hrs = prod.pivot_table(index=["MINE_ID", "year"], columns="subunit_cd", values="hours", aggfunc="sum", fill_value=0)
    total = hrs.sum(axis=1)
    emp = prod.groupby(["MINE_ID", "year"])["employees"].sum()
    a = flag_injuries(acc)
    op = a[~a["contractor"]]
    def count(mask, frame=op):
        return frame[mask].groupby(["MINE_ID", "year"]).size()
    df = pd.DataFrame({"hours": total, "employees": emp})
    df["share_underground"] = hrs.get("01", 0) / total.replace(0, np.nan)
    df["share_mill"] = hrs.get("30", 0) / total.replace(0, np.nan)
    df["share_pit"] = hrs.get("03", 0) / total.replace(0, np.nan)
    df["nfdl"] = count(op["nfdl"])
    df["all_injury"] = count(op["all_injury"])
    df["medical_only"] = count(op["degree"] == 6)
    df["contractor_injuries"] = a[a["contractor"] & a["all_injury"]].groupby(["MINE_ID", "year"]).size()
    df["new_worker_injuries"] = a[a["all_injury"] & (a["mine_exper"] < 1)].groupby(["MINE_ID", "year"]).size()
    df["days_lost"] = op[op["nfdl"]].groupby(["MINE_ID", "year"])["days"].sum()
    comp = mine_year(vio, ins)
    df = df.join(comp, how="left")
    df = df.reset_index()
    df["year"] = df["year"].astype(int)
    df = df[df["hours"] > 0].copy()
    cnt = ["nfdl", "all_injury", "medical_only", "contractor_injuries", "new_worker_injuries", "days_lost"] + list(comp.columns)
    df[cnt] = df[cnt].fillna(0)
    df = df.sort_values(["MINE_ID", "year"])
    g = df.groupby("MINE_ID")
    prev_year = g["year"].shift(1)
    consecutive = prev_year == df["year"] - 1
    df["nfdl_prev"] = g["nfdl"].shift(1).where(consecutive, 0).fillna(0)
    df["hours_growth"] = np.log(df["hours"] / g["hours"].shift(1).where(consecutive)).replace([np.inf, -np.inf], np.nan)
    df["years_reporting"] = g.cumcount()
    # three-year injury count on calendar years (missing years count as zero)
    idx = df.set_index(["MINE_ID", "year"])
    def three(col):
        s = idx[col]
        return [s.get((m, y), 0) + s.get((m, y - 1), 0) + s.get((m, y - 2), 0) for m, y in zip(df["MINE_ID"], df["year"])]
    df["nfdl_3yr"] = three("nfdl")
    df["hours_3yr"] = three("hours")
    rates = industry_rates(prod, acc)
    rate_of = lambda years: years.map(lambda y: rates.get(y, np.nan)).astype(float)
    df["industry_rate"] = rate_of(df["year"])
    df["relative_rate"] = (df["nfdl"] / df["hours"] * BASE_HOURS) / df["industry_rate"]
    exp_by = (df.set_index(["MINE_ID", "year"])["hours"] * df.set_index(["MINE_ID", "year"])["industry_rate"].to_numpy() / BASE_HOURS)
    df["expected_3yr"] = [exp_by.get((m, y), 0) + exp_by.get((m, y - 1), 0) + exp_by.get((m, y - 2), 0) for m, y in zip(df["MINE_ID"], df["year"])]
    df["relative_3yr"] = df["nfdl_3yr"] / df["expected_3yr"].replace(0, np.nan)
    df["log_hours"] = np.log(df["hours"])
    df["log_employees"] = np.log1p(df["employees"])
    df["log_penalties"] = np.log1p(df["penalties"])
    df["log_inspection_hours"] = np.log1p(df["inspection_hours"])
    df["violations_per_100h"] = 100 * df["violations"] / df["inspection_hours"].replace(0, np.nan)
    df["ss_per_100h"] = 100 * df["ss_violations"] / df["inspection_hours"].replace(0, np.nan)
    df["ss_share"] = df["ss_violations"] / df["violations"].replace(0, np.nan)
    # target: operator NFDL injuries next year, for mines still reporting hours next year
    nxt = df.set_index(["MINE_ID", "year"])
    key = list(zip(df["MINE_ID"], df["year"] + 1))
    df["next_hours"] = [nxt["hours"].get(k, np.nan) for k in key]
    df["next_nfdl"] = [nxt["nfdl"].get(k, np.nan) for k in key]
    df = df[(df["hours"] >= min_hours) & df["next_hours"].notna()]
    if last_year is not None:
        df = df[df["year"] + 1 <= last_year]
    df["exposure"] = df["next_hours"] / BASE_HOURS
    df["target_year"] = df["year"] + 1
    df["next_industry_rate"] = rate_of(df["target_year"])     # used only to scale training targets, never as an input
    return df.reset_index(drop=True)


MODELS = {
    "Industry rate": None,
    "Own recent rate, shrunk": None,
    "Poisson regression: size, mix and injury history": EXPOSURE + HISTORY,
    "Poisson regression: all inputs": EXPOSURE + HISTORY + COMPLIANCE,
    "Boosting: size, mix and injury history": EXPOSURE + HISTORY,
    "Boosting: all inputs": EXPOSURE + HISTORY + COMPLIANCE,
}


def _relative_target(tr: pd.DataFrame):
    """Next-year injuries against the count expected at that year's industry rate, and the expected count as weight
    (a Poisson model with offset log(expected))."""
    expected = tr["exposure"] * tr["next_industry_rate"]
    return tr["next_nfdl"] / expected, expected


def _fit_predict(name: str, cols: list[str], tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """Predicted injury rate per 200,000 hours: the predicted relative rate times the latest known industry rate."""
    y, w = _relative_target(tr)
    if name.startswith("Poisson"):
        med = tr[cols].median()
        m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.05, max_iter=3000))
        m.fit(tr[cols].fillna(med).fillna(0), y, poissonregressor__sample_weight=w)
        rel = m.predict(te[cols].fillna(med).fillna(0))
    else:
        rel = HistGradientBoostingRegressor(**GB_PARAMS).fit(tr[cols], y, sample_weight=w).predict(te[cols])
    return rel * te["industry_rate"].to_numpy()


def evaluate(df: pd.DataFrame, test_from: int = 2016) -> pd.DataFrame:
    """Rolling yearly test. Returns one row per test mine-year with each model's predicted rate."""
    out = []
    for ty in sorted(df.loc[df["target_year"] >= test_from, "target_year"].unique()):
        tr, te = df[df["target_year"] < ty], df[df["target_year"] == ty]
        if te.empty:
            continue
        r = te[["MINE_ID", "year", "target_year", "hours", "next_hours", "exposure", "nfdl", "next_nfdl", "industry_rate"]].copy()
        r["Industry rate"] = te["industry_rate"]
        r["Own recent rate, shrunk"] = te["industry_rate"] * (te["nfdl_3yr"] + PRIOR_INJURIES) / (te["expected_3yr"] + PRIOR_INJURIES)
        for name, cols in MODELS.items():
            if cols is not None:
                r[name] = _fit_predict(name, cols, tr, te)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-9, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu)
    return float(2 * np.mean(term))


def top_band(r: pd.DataFrame, col: str, frac: float = 0.2) -> dict:
    """Mines ranked highest each year by predicted rate: their share of next-year injuries and hours, and the rate ratio."""
    inj = hrs = 0.0
    for _, g in r.groupby("target_year"):
        k = max(1, int(round(frac * len(g))))
        top = g.nlargest(k, col)
        inj += top["next_nfdl"].sum()
        hrs += top["next_hours"].sum()
    inj_share, hrs_share = inj / r["next_nfdl"].sum(), hrs / r["next_hours"].sum()
    return {"injury_share_pct": 100 * inj_share, "hours_share_pct": 100 * hrs_share,
            "rate_ratio": float(inj_share / hrs_share) if hrs_share else float("nan")}


def score_table(r: pd.DataFrame) -> pd.DataFrame:
    y = r["next_nfdl"].to_numpy()
    d0 = poisson_deviance(y, (r["Industry rate"] * r["exposure"]).to_numpy())
    rows = {}
    for name in MODELS:
        d = poisson_deviance(y, (r[name] * r["exposure"]).to_numpy())
        tb = top_band(r, name) if name != "Industry rate" else {"injury_share_pct": np.nan, "hours_share_pct": np.nan, "rate_ratio": 1.0}
        rows[name] = {"deviance": d, "deviance_skill_pct": 100 * (1 - d / d0),
                      "top20_injury_share_pct": tb["injury_share_pct"], "top20_hours_share_pct": tb["hours_share_pct"],
                      "top20_rate_ratio": tb["rate_ratio"]}
    return pd.DataFrame(rows).T


def bootstrap_skill_gain(r: pd.DataFrame, a: str, b: str, n: int = 1000, seed: int = 0) -> dict:
    """Deviance skill of model a minus model b (percentage points), with a 90 percent interval from resampling
    whole mines (clustered bootstrap), so repeated years of the same mine are not treated as independent."""
    rng = np.random.default_rng(seed)
    def skill_pair(s):
        y = s["next_nfdl"].to_numpy()
        d0 = poisson_deviance(y, (s["Industry rate"] * s["exposure"]).to_numpy())
        da = poisson_deviance(y, (s[a] * s["exposure"]).to_numpy())
        db = poisson_deviance(y, (s[b] * s["exposure"]).to_numpy())
        return 100 * (db - da) / d0
    groups = [g for _, g in r.groupby("MINE_ID")]
    diffs = [skill_pair(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n)]
    return {"gain_points": skill_pair(r), "low": float(np.quantile(diffs, 0.05)), "high": float(np.quantile(diffs, 0.95)), "draws": n}


def grouped_importance(df: pd.DataFrame, test_from: int = 2016, repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Loss of deviance skill (points) for the all-inputs boosting model when one group of inputs is shuffled within each test year."""
    rng = np.random.default_rng(seed)
    cols = EXPOSURE + HISTORY + COMPLIANCE
    groups = {"Size and workforce mix": EXPOSURE, "Injury history": HISTORY, "Inspections and violations": COMPLIANCE}
    rows = {g: [] for g in groups}
    base_mu, ys, ind_mu, perm_mu = [], [], [], {g: [] for g in groups}
    for ty in sorted(df.loc[df["target_year"] >= test_from, "target_year"].unique()):
        tr, te = df[df["target_year"] < ty], df[df["target_year"] == ty]
        y_rel, w = _relative_target(tr)
        m = HistGradientBoostingRegressor(**GB_PARAMS).fit(tr[cols], y_rel, sample_weight=w)
        e = (te["exposure"] * te["industry_rate"]).to_numpy()
        base_mu.append(m.predict(te[cols]) * e)
        ind_mu.append(e)
        ys.append(te["next_nfdl"].to_numpy())
        for g, gc in groups.items():
            reps = []
            for _ in range(repeats):
                Xp = te[cols].copy()
                Xp[gc] = Xp[gc].to_numpy()[rng.permutation(len(Xp))]
                reps.append(m.predict(Xp) * e)
            perm_mu[g].append(np.array(reps))
    y, mu0, mui = np.concatenate(ys), np.concatenate(base_mu), np.concatenate(ind_mu)
    d0, dind = poisson_deviance(y, mu0), poisson_deviance(y, mui)
    for g in groups:
        arr = np.concatenate(perm_mu[g], axis=1)
        rows[g] = [100 * (poisson_deviance(y, arr[k]) - d0) / dind for k in range(arr.shape[0])]
    return pd.DataFrame({"skill_loss_points": {g: float(np.mean(v)) for g, v in rows.items()},
                         "spread": {g: float(np.std(v)) for g, v in rows.items()}}).sort_values("skill_loss_points", ascending=False)


def risk_bands(r: pd.DataFrame, col: str = "Boosting: size, mix and injury history", bands: int = 5) -> pd.DataFrame:
    """Mines grouped each year into bands by predicted rate; observed next-year rate per band, pooled over test years."""
    parts = []
    for _, g in r.groupby("target_year"):
        g = g.copy()
        g["band"] = pd.qcut(g[col].rank(method="first"), bands, labels=[f"Band {i}" for i in range(1, bands + 1)])
        parts.append(g)
    a = pd.concat(parts)
    t = a.groupby("band", observed=True).agg(mine_years=("next_nfdl", "size"), injuries=("next_nfdl", "sum"),
                                             exposure=("exposure", "sum"),
                                             predicted_rate=(col, lambda s: float(np.average(s, weights=a.loc[s.index, "exposure"]))))
    t["observed_rate"] = t["injuries"] / t["exposure"]
    t["hours_m"] = t.pop("exposure") * BASE_HOURS / 1e6
    return t


def by_year(r: pd.DataFrame, models=("Own recent rate, shrunk", "Boosting: size, mix and injury history", "Boosting: all inputs")) -> pd.DataFrame:
    rows = {}
    for ty, g in r.groupby("target_year"):
        y = g["next_nfdl"].to_numpy()
        d0 = poisson_deviance(y, (g["Industry rate"] * g["exposure"]).to_numpy())
        rows[int(ty)] = {"mines": len(g), "injuries": int(y.sum()),
                         **{m: 100 * (1 - poisson_deviance(y, (g[m] * g["exposure"]).to_numpy()) / d0) for m in models}}
    return pd.DataFrame(rows).T


def terciles(r: pd.DataFrame, col: str = "Own recent rate, shrunk") -> pd.DataFrame:
    """Each test year's mines split into thirds by predicted rate; predicted and observed rates weighted by hours."""
    rows = []
    for ty, g in r.groupby("target_year"):
        q = pd.qcut(g[col].rank(method="first"), 3, labels=["best third", "middle third", "worst third"])
        for lab, h in g.groupby(q, observed=True):
            rows.append({"target_year": int(ty), "third": str(lab), "mines": len(h),
                         "predicted_rate": float(np.average(h[col], weights=h["exposure"])),
                         "observed_rate": float(h["next_nfdl"].sum() / h["exposure"].sum())})
    return pd.DataFrame(rows)
