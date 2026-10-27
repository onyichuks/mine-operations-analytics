"""MSHA inspections and violations at gold ore mines.

S&S (significant and substantial) is the inspector's finding that a violation is reasonably likely to lead to a
reasonably serious injury or illness. Orders are the most serious enforcement actions (for example withdrawal of
miners from an area). Penalties are the proposed amounts before any settlement, in dollars of the year issued.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Title 30 Code of Federal Regulations. Part 56 covers surface metal and nonmetal mines, Part 57 underground
# metal and nonmetal mines (and their surface areas), Part 47 hazard communication, Part 50 reporting.
STANDARDS = {
    "14100": "Safety defects on equipment: examination, correction and records",
    "3360": "Ground support use",
    "3200": "Correction of hazardous ground conditions",
    "47.41": "Hazard communication: container labels",
    "11001": "Safe access to working places",
    "20003": "Housekeeping",
    "12004": "Electrical conductors protected and insulated",
    "12028": "Testing grounding systems",
    "4201": "Inspection of fire extinguishers",
    "50.30": "Quarterly employment report (MSHA Form 7000-2)",
    "4104": "Combustible waste",
    "14107": "Guarding moving machine parts",
    "4102": "Spillage and leakage of flammable or combustible liquids",
    "20011": "Barricades and warning signs",
    "12032": "Inspection and cover plates on electrical equipment",
    "11051": "Escape routes",
    "14132": "Horns and backup alarms",
}


def standard_code(part_section: pd.Series) -> pd.Series:
    """'56.14100(b)' -> '56.14100' (the paragraph letter is dropped so surface and underground forms group cleanly)."""
    return part_section.fillna("").str.strip().str.replace(r"\(.*$", "", regex=True)


def describe(code: str) -> str:
    if code in STANDARDS:
        return STANDARDS[code]
    sec = code.split(".", 1)[-1]
    return STANDARDS.get(sec, "")


def activity_group(activity: pd.Series) -> pd.Series:
    a = activity.fillna("").str.upper()
    out = pd.Series("Other", index=activity.index)
    out[a.str.contains("REGULAR")] = "Regular inspection"
    out[a.str.contains("FOLLOW-UP")] = "Compliance follow-up"
    out[a.str.contains("SPOT")] = "Spot inspection"
    out[a.str.contains("COMPLAINT")] = "Hazard complaint"
    out[a.str.contains("ACCIDENT INVESTIGATION")] = "Accident investigation"
    return out


def annual(vio: pd.DataFrame, ins: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    years = range(first, last + 1)
    v = vio[vio["year"].between(first, last)]
    i = ins[ins["year"].between(first, last)].copy()
    i["group"] = activity_group(i["ACTIVITY"])
    t = pd.DataFrame(index=pd.Index(years, name="year"))
    t["inspections"] = i.groupby("year").size().reindex(years, fill_value=0)
    t["regular_inspections"] = i[i["group"] == "Regular inspection"].groupby("year").size().reindex(years, fill_value=0)
    t["complaint_inspections"] = i[i["group"] == "Hazard complaint"].groupby("year").size().reindex(years, fill_value=0)
    t["inspection_hours"] = i.groupby("year")["insp_hours"].sum().reindex(years, fill_value=0)
    t["violations"] = v.groupby("year").size().reindex(years, fill_value=0)
    t["contractor_violations"] = v[v["contractor"]].groupby("year").size().reindex(years, fill_value=0)
    t["ss_violations"] = v[v["ss"]].groupby("year").size().reindex(years, fill_value=0)
    t["orders"] = v[v["order"]].groupby("year").size().reindex(years, fill_value=0)
    t["high_negligence"] = v[v["high_negligence"]].groupby("year").size().reindex(years, fill_value=0)
    t["proposed_penalties_k"] = v.groupby("year")["penalty"].sum().reindex(years, fill_value=0) / 1000
    t["ss_share_pct"] = 100 * t["ss_violations"] / t["violations"].replace(0, np.nan)
    t["violations_per_100_insp_hours"] = 100 * t["violations"] / t["inspection_hours"].replace(0, np.nan)
    t["ss_per_100_insp_hours"] = 100 * t["ss_violations"] / t["inspection_hours"].replace(0, np.nan)
    return t


def top_standards(vio: pd.DataFrame, years: range, top: int = 15) -> pd.DataFrame:
    v = vio[vio["year"].isin(years)].copy()
    v["standard"] = standard_code(v["PART_SECTION"])
    t = v.groupby("standard").agg(citations=("ss", "size"), ss_share=("ss", "mean"), penalties=("penalty", "sum"))
    t["ss_share_pct"] = 100 * t.pop("ss_share")
    t["share_of_all_pct"] = 100 * t["citations"] / len(v)
    t = t.sort_values("citations", ascending=False).head(top)
    t["description"] = [describe(c) for c in t.index]
    return t


def activity_mix(ins: pd.DataFrame, years: range) -> pd.DataFrame:
    i = ins[ins["year"].isin(years)].copy()
    i["group"] = activity_group(i["ACTIVITY"])
    t = i.groupby("group").agg(inspections=("EVENT_NO", "size"), hours=("insp_hours", "sum"))
    t["share_pct"] = 100 * t["inspections"] / t["inspections"].sum()
    return t.sort_values("inspections", ascending=False)


def mine_year(vio: pd.DataFrame, ins: pd.DataFrame) -> pd.DataFrame:
    """Compliance indicators for each mine and year (input to the risk model)."""
    v = vio.dropna(subset=["year"])
    g = v.groupby(["MINE_ID", "year"])
    out = pd.DataFrame({
        "violations": g.size(),
        "ss_violations": g["ss"].sum(),
        "orders": g["order"].sum(),
        "high_negligence": g["high_negligence"].sum(),
        "penalties": g["penalty"].sum(),
        "contractor_violations": g["contractor"].sum(),
    })
    i = ins.dropna(subset=["year"]).copy()
    i["group"] = activity_group(i["ACTIVITY"])
    gi = i.groupby(["MINE_ID", "year"])
    ii = pd.DataFrame({
        "inspections": gi.size(),
        "inspection_hours": gi["insp_hours"].sum(),
        "complaint_inspections": gi["group"].apply(lambda s: int((s == "Hazard complaint").sum())),
    })
    out = out.join(ii, how="outer").fillna(0)
    out.index = out.index.set_names(["MINE_ID", "year"])
    return out
