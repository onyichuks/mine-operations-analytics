"""Gold mine safety and compliance: injury rates, injury patterns, inspections and violations, and the injury rate model.

Usage:
    python scripts/get_data.py && python scripts/prepare_data.py     # once
    python run_safety.py                    # under a minute; writes outputs/safety_*.json and outputs/tables/s*.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mineops import compliance, data, risk, safety  # noqa: E402
from mineops.outputs import rnd, to_json  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
RECENT = 10   # years in the "recent" window used for patterns


def data_checks(g: dict, last: int) -> dict:
    prod, acc, ins, vio, mines = g["prod"], g["accidents"], g["inspections"], g["violations"], g["mines"]
    mine_years = set(zip(prod.loc[prod["hours"] > 0, "MINE_ID"], prod.loc[prod["hours"] > 0, "year"]))
    op_inj = acc[(~acc["contractor"]) & acc["degree"].isin(safety.ALL_INJURY) & acc["year"].le(last)]
    no_hours = sum((m, y) not in mine_years for m, y in zip(op_inj["MINE_ID"], op_inj["year"]))
    nfdl = acc[acc["degree"].isin(safety.NFDL)]
    return {
        "gold_mines": int(len(mines)),
        "mines_by_status": mines["CURRENT_MINE_STATUS"].value_counts().to_dict(),
        "mines_by_type": mines["CURRENT_MINE_TYPE"].value_counts().to_dict(),
        "mines_by_state_top": mines["STATE"].value_counts().head(8).to_dict(),
        "records": {"employment_rows": int(len(prod)), "accidents": int(len(acc)), "inspections": int(len(ins)), "violations": int(len(vio))},
        "years": [int(acc["year"].min()), int(acc["year"].max())],
        "last_full_year": last,
        "partial_year_hours_m": float(prod.loc[prod["year"] == last + 1, "hours"].sum() / 1e6),
        "duplicate_accident_rows": int(acc.duplicated().sum()),
        "duplicate_violation_numbers": int(vio["VIOLATION_NO"].duplicated().sum()),
        "accidents_degree_missing": int(acc["degree"].isna().sum()),
        "operator_injuries_without_hours_that_year": int(no_hours),
        "operator_injuries_checked": int(len(op_inj)),
        "nfdl_mine_experience_missing_pct": float(100 * nfdl["mine_exper"].isna().mean()),
        "inspections_without_hours": int((ins["insp_hours"] <= 0).sum()),
        "contractor_records": int(acc["contractor"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--min-hours", type=float, default=10_000, help="smallest mine (employee hours in a year) in the rate model")
    args = ap.parse_args()
    out = Path(args.out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    print("1/5 Loading and checking the MSHA gold mine records")
    g = data.load_gold(Path(args.data))
    prod, acc, ins, vio = g["prod"], g["accidents"], g["inspections"], g["violations"]
    last = safety.last_full_year(prod, acc)
    recent = range(last - RECENT + 1, last + 1)
    checks = data_checks(g, last)

    print("2/5 Injury rates and patterns")
    ann = safety.annual(prod, acc, last=last)
    ann.round(4).to_csv(tables / "s01_annual_injury_rates.csv")
    sub = safety.by_subunit(prod, acc, recent)
    sub.rename_axis("subunit_code").round(4).to_csv(tables / "s02_rates_by_subunit_recent.csv")
    sub_tr = safety.subunit_trend(prod, acc, 2000, last)
    sub_tr.round(4).to_csv(tables / "s03_nfdl_rate_by_subunit_and_year.csv")
    cls = safety.classifications(acc, recent)
    cls.rename_axis("classification").round(2).to_csv(tables / "s04_lost_time_injury_classifications_recent.csv")
    exp = safety.experience(acc, recent)
    exp_all = {int(k): v for k, v in exp["under_1_year_by_year"].items() if int(k) <= last}
    fat = safety.fatalities(acc, 2000, last)
    fat.rename_axis("classification").to_csv(tables / "s05_fatalities_by_classification.csv")
    parts = safety.body_parts_and_activity(acc, recent)

    print("3/5 Inspections and violations")
    comp = compliance.annual(vio, ins, 2000, last)
    comp.round(3).to_csv(tables / "s06_inspections_and_violations_by_year.csv")
    top = compliance.top_standards(vio, recent)
    top.rename(index=lambda c: f"30 CFR {c}").rename_axis("standard").round(2).to_csv(tables / "s07_most_cited_standards_recent.csv")
    mix = compliance.activity_mix(ins, recent)
    mix.rename_axis("activity").round(2).to_csv(tables / "s08_inspection_activity_mix_recent.csv")

    print("4/5 Injury rate model (rolling yearly test)")
    panel = risk.panel(prod, acc, vio, ins, min_hours=args.min_hours, last_year=last)
    test_from = last - 9
    res = risk.evaluate(panel, test_from=test_from)
    st = risk.score_table(res)
    st.rename_axis("model").round(6).to_csv(tables / "s09_rate_model_scores.csv")
    gain_comp = risk.bootstrap_skill_gain(res, "Boosting: all inputs", "Boosting: size, mix and injury history")
    gain_ml = risk.bootstrap_skill_gain(res, "Boosting: size, mix and injury history", "Own recent rate, shrunk")
    gain_poisson = risk.bootstrap_skill_gain(res, "Poisson regression: all inputs", "Poisson regression: size, mix and injury history")
    thirds = risk.terciles(res)
    thirds.round(4).to_csv(tables / "s13_rate_model_thirds_by_year.csv", index=False)
    imp = risk.grouped_importance(panel, test_from=test_from)
    imp.rename_axis("input_group").round(4).to_csv(tables / "s10_rate_model_grouped_importance.csv")
    bands = risk.risk_bands(res)
    bands.round(4).to_csv(tables / "s11_rate_model_bands.csv")
    yearly = risk.by_year(res)
    yearly.rename_axis("target_year").round(3).to_csv(tables / "s12_rate_model_skill_by_year.csv")

    print("5/5 Writing results")
    a = ann
    first_rate, last_rate = float(a.loc[2000, "all_injury_rate"]), float(a.loc[last, "all_injury_rate"])
    results = {
        "data": checks,
        "definitions": {"base_hours": safety.BASE_HOURS, "all_injury_degree_codes": sorted(safety.ALL_INJURY),
                        "nfdl_degree_codes": sorted(safety.NFDL), "rates": "mine operator employees only"},
        "trend": {
            "all_injury_rate_2000": first_rate, f"all_injury_rate_{last}": last_rate,
            "nfdl_rate_2000": float(a.loc[2000, "nfdl_rate"]), f"nfdl_rate_{last}": float(a.loc[last, "nfdl_rate"]),
            "nfdl_rate_min": float(a["nfdl_rate"].min()), "nfdl_rate_min_year": int(a["nfdl_rate"].idxmin()),
            "operator_hours_m_last": float(a.loc[last, "operator_hours_m"]),
            "mines_reporting_last": int(a.loc[last, "mines_reporting_hours"]),
            "fatalities_2000_to_last": int(a["operator_fatal"].sum() + a["contractor_fatal"].sum()),
            "fatalities_last_10_years": int(a.loc[last - 9:last, ["operator_fatal", "contractor_fatal"]].sum().sum()),
            "severity": safety.severity(acc, [(last - 19, last - 10), (last - 9, last - 5), (last - 4, last)]),
            "contractor_share_of_injuries_recent_pct": float(100 * a.loc[recent, "contractor_all_injury"].sum()
                                                            / a.loc[recent, ["contractor_all_injury", "operator_all_injury"]].sum().sum()),
        },
        "subunits_recent": sub.drop(columns=[]).to_dict(orient="index"),
        "classifications_recent": cls.to_dict(orient="index"),
        "experience_recent": {k: v for k, v in exp.items() if k != "under_1_year_by_year"},
        "fatalities": fat.to_dict(orient="index"),
        "patterns_recent": parts,
        "compliance": {
            "recent_years": [recent.start, recent.stop - 1],
            "ss_share_pct_by_year": comp["ss_share_pct"].round(2).to_dict(),
            "violations_per_100_insp_hours_recent": float(100 * comp.loc[recent, "violations"].sum() / comp.loc[recent, "inspection_hours"].sum()),
            "top_standards": top.to_dict(orient="index"),
            "activity_mix": mix.to_dict(orient="index"),
        },
        "rate_model": {
            "min_hours": args.min_hours, "test_years": [test_from, last],
            "panel_rows": int(len(panel)), "test_rows": int(len(res)), "test_injuries": int(res["next_nfdl"].sum()),
            "scores": st.to_dict(orient="index"),
            "gain_from_compliance_points": gain_comp,
            "gain_from_boosting_over_own_rate_points": gain_ml,
            "gain_from_compliance_poisson_points": gain_poisson,
            "thirds_last_year": thirds[thirds["target_year"] == last].to_dict(orient="records"),
            "importance": imp["skill_loss_points"].to_dict(),
            "bands": bands.to_dict(orient="index"),
            "skill_by_year": yearly.to_dict(orient="index"),
        },
    }
    (out / "safety_results.json").write_text(to_json(results, indent=2))

    yrs = a.index.tolist()
    dash = {
        "checks": checks, "last": last, "recent": [recent.start, recent.stop - 1],
        "severity": results["trend"]["severity"], "trend": {k: v for k, v in results["trend"].items() if k != "severity"},
        "annual": {"year": yrs, **{c: rnd(a[c], 4) for c in a.columns}},
        "subunit": {"name": sub["subunit"].tolist(), **{c: rnd(sub[c], 4) for c in ["hours_m", "all_injury", "nfdl", "fatal", "all_injury_rate", "nfdl_rate"]}},
        "subunit_trend": {"year": sub_tr.index.tolist(), **{c: rnd(sub_tr[c], 3) for c in sub_tr.columns}},
        "classes": {"name": cls.index.tolist(), **{c: rnd(cls[c], 2) for c in cls.columns}},
        "experience": {**{k: v for k, v in exp.items() if k != "under_1_year_by_year"}, "by_year": exp_all},
        "fatalities": {"name": fat.index.tolist(), "n": fat["fatalities"].tolist(), "contractor": fat["contractor"].tolist()},
        "patterns": parts,
        "compliance": {"year": comp.index.tolist(), **{c: rnd(comp[c], 3) for c in comp.columns}},
        "standards": {"code": top.index.tolist(), "description": top["description"].tolist(), **{c: rnd(top[c], 2) for c in ["citations", "ss_share_pct", "share_of_all_pct"]}},
        "mix": {"group": mix.index.tolist(), "inspections": mix["inspections"].tolist(), "share": rnd(mix["share_pct"], 1)},
        "model": {
            "scores": {k: {kk: rnd(vv, 4) for kk, vv in v.items()} for k, v in st.to_dict(orient="index").items()},
            "gain_comp": gain_comp, "gain_ml": gain_ml, "gain_poisson": gain_poisson,
            "thirds_last": {r["third"]: {"predicted": r["predicted_rate"], "observed": r["observed_rate"]} for r in thirds[thirds["target_year"] == last].to_dict(orient="records")},
            "importance": {"group": imp.index.tolist(), "loss": rnd(imp["skill_loss_points"], 2)},
            "bands": {"band": bands.index.astype(str).tolist(), "observed": rnd(bands["observed_rate"], 3),
                      "predicted": rnd(bands["predicted_rate"], 3), "hours_m": rnd(bands["hours_m"], 1), "mine_years": bands["mine_years"].tolist()},
            "by_year": {"year": [int(i) for i in yearly.index], **{c: rnd(yearly[c], 2) for c in yearly.columns}},
            "test_years": [test_from, last], "test_rows": int(len(res)), "min_hours": args.min_hours,
        },
    }
    (out / "safety_dashboard.json").write_text(to_json(dash, compact=True))
    t = results["trend"]
    m = results["rate_model"]["scores"]
    print(f"Operator injury rate per 200,000 hours: {first_rate:.2f} (2000) to {last_rate:.2f} ({last}); NFDL {t['nfdl_rate_2000']:.2f} to {t[f'nfdl_rate_{last}']:.2f}")
    print(f"Rate model skill: own shrunk rate {m['Own recent rate, shrunk']['deviance_skill_pct']:.1f}%, boosting {m['Boosting: size, mix and injury history']['deviance_skill_pct']:.1f}%, "
          f"with inspections and violations {m['Boosting: all inputs']['deviance_skill_pct']:.1f}%")


if __name__ == "__main__":
    main()
