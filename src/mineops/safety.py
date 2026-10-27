"""Injury rates and injury patterns at United States gold ore mines (MSHA Part 50 reports).

Definitions
* Rates are per 200,000 employee hours (about 100 full-time workers for a year), as MSHA publishes them.
* Rates use mine operator employees only. MSHA reports contractor hours as company totals, not by mine, so
  contractor injuries are counted but no contractor rate is calculated.
* All injuries: degree of injury codes 1 to 6 (fatal, permanent disability, days away, days away and restricted,
  restricted only, and medical treatment with no days lost). Accidents without injury (0), occupational illness (7),
  natural causes (8), injuries to non-employees (9) and other cases (10) are left out.
* NFDL (non-fatal, days lost): codes 2 to 5, the definition NIOSH uses for lost-time injuries.
* The latest year is incomplete (hours run to mid-year), so trends stop at the last full year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_HOURS = 200_000
ALL_INJURY = {1, 2, 3, 4, 5, 6}
NFDL = {2, 3, 4, 5}
FATAL = {1}
SUBUNITS = {"01": "Underground", "02": "Surface at underground mine", "03": "Open pit", "06": "Dredge",
            "12": "Other mining", "30": "Mill or preparation plant", "99": "Office"}
MAIN_SUBUNITS = ["03", "01", "30", "02"]


def last_full_year(prod: pd.DataFrame, acc: pd.DataFrame) -> int:
    """The most recent year is reported part-way through; treat the year before the latest as the last full year."""
    return int(min(prod["year"].max(), acc["year"].max())) - 1


def flag_injuries(acc: pd.DataFrame) -> pd.DataFrame:
    a = acc.copy()
    a["all_injury"] = a["degree"].isin(ALL_INJURY)
    a["nfdl"] = a["degree"].isin(NFDL)
    a["fatal"] = a["degree"].isin(FATAL)
    a["days"] = a["days_lost"].fillna(0) + a["days_restrict"].fillna(0)
    return a


def annual(prod: pd.DataFrame, acc: pd.DataFrame, first: int = 2000, last: int | None = None) -> pd.DataFrame:
    a = flag_injuries(acc)
    last = last or last_full_year(prod, acc)
    years = range(first, last + 1)
    op = a[~a["contractor"]]
    co = a[a["contractor"]]
    hrs = prod.groupby("year")["hours"].sum()
    mines = prod[prod["hours"] > 0].groupby("year")["MINE_ID"].nunique()
    emp = prod.groupby("year")["employees"].sum()
    t = pd.DataFrame(index=pd.Index(years, name="year"))
    t["operator_hours_m"] = hrs.reindex(years) / 1e6
    t["mines_reporting_hours"] = mines.reindex(years)
    t["avg_employees"] = emp.reindex(years)
    for k in ("all_injury", "nfdl", "fatal"):
        t[f"operator_{k}"] = op[op[k]].groupby("year").size().reindex(years, fill_value=0)
        t[f"contractor_{k}"] = co[co[k]].groupby("year").size().reindex(years, fill_value=0)
    t["all_injury_rate"] = t["operator_all_injury"] / (t["operator_hours_m"] * 1e6) * BASE_HOURS
    t["nfdl_rate"] = t["operator_nfdl"] / (t["operator_hours_m"] * 1e6) * BASE_HOURS
    t["contractor_share_of_injuries_pct"] = 100 * t["contractor_all_injury"] / (t["contractor_all_injury"] + t["operator_all_injury"])
    t["days_per_nfdl"] = op[op["nfdl"]].groupby("year")["days"].median().reindex(years)
    return t


def by_subunit(prod: pd.DataFrame, acc: pd.DataFrame, years: range) -> pd.DataFrame:
    a = flag_injuries(acc)
    op = a[~a["contractor"] & a["year"].isin(years)]
    p = prod[prod["year"].isin(years)]
    hrs = p.groupby("subunit_cd")["hours"].sum()
    t = pd.DataFrame({"hours_m": hrs / 1e6})
    t["all_injury"] = op[op["all_injury"]].groupby("subunit_cd").size()
    t["nfdl"] = op[op["nfdl"]].groupby("subunit_cd").size()
    t["fatal"] = op[op["fatal"]].groupby("subunit_cd").size()
    t = t.fillna(0)
    t["all_injury_rate"] = t["all_injury"] / (t["hours_m"] * 1e6) * BASE_HOURS
    t["nfdl_rate"] = t["nfdl"] / (t["hours_m"] * 1e6) * BASE_HOURS
    t["subunit"] = [SUBUNITS.get(c, c) for c in t.index]
    return t.sort_values("hours_m", ascending=False)


def subunit_trend(prod: pd.DataFrame, acc: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """NFDL rate by year for the main subunits."""
    a = flag_injuries(acc)
    op = a[~a["contractor"] & a["nfdl"]]
    rows = {}
    for cd in MAIN_SUBUNITS:
        hrs = prod[prod["subunit_cd"] == cd].groupby("year")["hours"].sum()
        n = op[op["subunit_cd"] == cd].groupby("year").size()
        yrs = range(first, last + 1)
        rows[SUBUNITS[cd]] = (n.reindex(yrs, fill_value=0) / hrs.reindex(yrs) * BASE_HOURS)
    return pd.DataFrame(rows).rename_axis("year")


def classifications(acc: pd.DataFrame, years: range, top: int = 10) -> pd.DataFrame:
    """What lost-time injuries come from, overall and by subunit, with the median days lost or restricted."""
    a = flag_injuries(acc)
    op = a[~a["contractor"] & a["nfdl"] & a["year"].isin(years)]
    total = op.groupby("CLASSIFICATION").agg(nfdl=("nfdl", "size"), median_days=("days", "median"))
    total["share_pct"] = 100 * total["nfdl"] / total["nfdl"].sum()
    for cd in ["03", "01", "30"]:
        sub = op[op["subunit_cd"] == cd]
        total[f"share_{SUBUNITS[cd].split()[0].lower()}_pct"] = 100 * sub.groupby("CLASSIFICATION").size() / max(1, len(sub))
    total = total.sort_values("nfdl", ascending=False).fillna(0)
    head = total.head(top)
    rest = total.iloc[top:]
    if len(rest):
        other = pd.DataFrame([{"nfdl": rest["nfdl"].sum(), "median_days": np.nan, "share_pct": rest["share_pct"].sum(),
                               **{c: rest[c].sum() for c in total.columns if c.startswith("share_") and c != "share_pct"}}],
                             index=["All other classifications"])
        head = pd.concat([head, other])
    head.index = [str(i).capitalize() for i in head.index]
    return head


def experience(acc: pd.DataFrame, years: range) -> dict:
    """How new to the mine the injured workers were (operator and contractor injuries with days lost)."""
    a = flag_injuries(acc)
    d = a[a["nfdl"] & a["year"].isin(years)]
    me = d["mine_exper"].dropna()
    te = d["tot_exper"].dropna()
    bins = [0, 0.25, 1, 3, 5, 10, 100]
    labels = ["under 3 months", "3 to 12 months", "1 to 3 years", "3 to 5 years", "5 to 10 years", "10 years or more"]
    dist = pd.cut(me, bins=bins, labels=labels, right=False).value_counts().reindex(labels, fill_value=0)
    known = a[a["nfdl"] & a["mine_exper"].notna()]
    by_year = (known["mine_exper"] < 1).groupby(known["year"]).mean() * 100
    return {
        "cases_with_mine_experience": int(len(me)),
        "under_1_year_at_mine_pct": float(100 * (me < 1).mean()),
        "under_3_months_at_mine_pct": float(100 * (me < 0.25).mean()),
        "under_1_year_in_mining_pct": float(100 * (te < 1).mean()),
        "median_years_at_mine": float(me.median()),
        "distribution": {k: int(v) for k, v in dist.items()},
        "contractor_under_1_year_pct": float(100 * (d.loc[d["contractor"], "mine_exper"].dropna() < 1).mean()),
        "operator_under_1_year_pct": float(100 * (d.loc[~d["contractor"], "mine_exper"].dropna() < 1).mean()),
        "under_1_year_by_year": {int(k): float(v) for k, v in by_year.items()},
    }


def fatalities(acc: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    a = flag_injuries(acc)
    f = a[a["fatal"] & a["year"].between(first, last)]
    t = f.groupby(["CLASSIFICATION"]).agg(fatalities=("fatal", "size"), contractor=("contractor", "sum"))
    t.index = [str(i).capitalize() for i in t.index]
    return t.sort_values("fatalities", ascending=False)


def body_parts_and_activity(acc: pd.DataFrame, years: range, top: int = 8) -> dict:
    a = flag_injuries(acc)
    d = a[~a["contractor"] & a["nfdl"] & a["year"].isin(years)]
    def share(col):
        s = d[col].fillna("Not recorded").str.strip().str.capitalize().value_counts()
        return {k: float(100 * v / s.sum()) for k, v in s.head(top).items()}
    return {"accident_type": share("ACCIDENT_TYPE"), "body_part": share("INJ_BODY_PART"),
            "nature": share("NATURE_INJURY"), "occupation": share("OCCUPATION")}


def severity(acc: pd.DataFrame, periods: list[tuple[int, int]]) -> dict:
    """Days away or restricted per lost-time injury (operator employees), pooled over each period."""
    a = flag_injuries(acc)
    d = a[~a["contractor"] & a["nfdl"]]
    out = {}
    for lo, hi in periods:
        x = d.loc[d["year"].between(lo, hi), "days"]
        out[f"{lo}-{hi}"] = {"cases": int(len(x)), "median_days": float(x.median()), "mean_days": float(x.mean()),
                             "share_over_30_days_pct": float(100 * (x > 30).mean())}
    return out
