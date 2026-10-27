"""Processing plant: data trust audit, KPIs and the silica soft sensor.

Usage:
    python scripts/get_data.py && python scripts/prepare_data.py     # once
    python run_plant.py                     # about two minutes; writes outputs/plant_*.json and outputs/tables/p*.csv
    python run_plant.py --delay 2           # laboratory turnaround assumed for the central case (hours)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mineops import data, plant_kpis, plant_quality, softsensor  # noqa: E402
from mineops.outputs import rnd, to_json  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--delay", type=int, default=2, help="laboratory turnaround in hours for the central case")
    args = ap.parse_args()
    out = Path(args.out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    print("1/6 Loading and auditing the plant data")
    df = data.load_flotation(Path(args.data))
    h = plant_quality.hourly(df)
    lab = plant_quality.classify_lab(h)
    audit = plant_quality.audit(df, h, lab)
    stuck = plant_quality.stuck_periods(df)
    frozen = plant_quality.frozen_hours(h)
    hc = plant_quality.clean_hourly(h, stuck, frozen)
    lab.value_counts().reindex(plant_quality.LAB_CLASSES, fill_value=0).rename("hours").rename_axis("silica_value_class").to_csv(tables / "p01_lab_value_classes.csv")
    pd.DataFrame(audit["flatlines"]).to_csv(tables / "p02_stuck_sensor_screen.csv", index=False)

    print("2/6 Plant KPIs")
    monthly = plant_kpis.monthly(hc, lab)
    monthly.rename_axis("month").round(4).to_csv(tables / "p03_monthly_kpis.csv")
    daily = plant_kpis.daily(hc, lab)
    cond = plant_kpis.off_spec_conditions(hc, lab)
    cond.rename_axis("tag").round(4).to_csv(tables / "p04_off_spec_conditions.csv")
    resp = plant_kpis.lagged_response(hc, lab)
    resp.round(4).to_csv(tables / "p05_operator_response_lagged_correlation.csv")
    kpi = plant_kpis.summary(hc, lab)

    print("3/6 Tuning on May (calibration month) only")
    grid = softsensor.tune(hc, lab, args.delay, frozen)
    grid.round(4).to_csv(tables / "p06_tuning_grid.csv", index=False)
    best = grid.iloc[0]
    params = dict(softsensor.GB_PARAMS, learning_rate=float(best["learning_rate"]), max_leaf_nodes=int(best["max_leaf_nodes"]),
                  min_samples_leaf=int(best["min_samples_leaf"]), l2_regularization=float(best["l2_regularization"]))

    print(f"4/6 Soft sensor, rolling monthly test, {args.delay}-hour laboratory turnaround")
    preds, models = softsensor.rolling_eval(hc, lab, args.delay, frozen, params=params, classify=True, keep_models=True)
    sc = softsensor.scores(preds)
    sc.rename_axis("test_month").round(6).to_csv(tables / "p07_soft_sensor_scores.csv")
    warn = softsensor.warning_scores(preds)
    imp = softsensor.grouped_importance(hc, lab, args.delay, frozen, models, preds)
    imp.rename_axis("input_group").round(6).to_csv(tables / "p08_grouped_importance.csv")

    print("5/6 Laboratory turnaround sweep, learning and training curves, leakage check")
    sweep = softsensor.delay_sweep(hc, lab, frozen, params=params)
    sweep.round(4).to_csv(tables / "p09_turnaround_sweep.csv")
    lc = softsensor.learning_curve(hc, lab, args.delay, frozen, params=params)
    lc.round(4).to_csv(tables / "p10_learning_curve.csv")
    tc = softsensor.training_curve(hc, lab, args.delay, frozen, params=params)
    tc.round(4).to_csv(tables / "p11_training_curve.csv")
    leak = softsensor.leakage_demo(df, hc, lab, sc.loc["all"], frozen=frozen, params=params, delay=args.delay)
    leak.round(4).to_csv(tables / "p12_leakage_check.csv", index=False)

    print("6/6 Writing results")
    allsc = sc.loc["all"]
    results = {
        "data": audit,
        "kpis": kpi,
        "soft_sensor": {
            "lab_turnaround_hours": args.delay,
            "test_period": [str(preds.index.min()), str(preds.index.max())],
            "test_hours": int(len(preds)),
            "params": {k: params[k] for k in ("learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_iter")},
            "mae": {k: float(allsc[f"{k}_mae"]) for k in softsensor.POINT_MODELS},
            "r2": {k: float(allsc[f"{k}_r2"]) for k in softsensor.POINT_MODELS},
            "gain_vs_persistence_pct": float(100 * (1 - allsc["shrunk_mae"] / allsc["persistence_mae"])),
            "coverage_80": float(allsc["coverage_80"]),
            "interval_width": float(allsc["interval_width"]),
            "weights_by_month": {m: float(sc.loc[m, "weight"]) for m in sc.index if m != "all"},
            "by_month": {m: {"persistence_mae": float(sc.loc[m, "persistence_mae"]), "shrunk_mae": float(sc.loc[m, "shrunk_mae"]),
                             "coverage_80": float(sc.loc[m, "coverage_80"])} for m in sc.index if m != "all"},
            "warning": warn,
            "importance": imp["mae_increase"].to_dict(),
            "turnaround_sweep": sweep.reset_index().to_dict(orient="records"),
            "leakage": leak.to_dict(orient="records"),
        },
    }
    (out / "plant_results.json").write_text(to_json(results, indent=2))

    ts = h["silica_conc"]
    dash = {
        "kpi": kpi,
        "audit": {k: audit[k] for k in ("rows", "period", "hours_in_period", "hours_with_data", "gaps", "lab_classes",
                                         "measured_share_pct", "feed_assay_update_hours", "iron_silica_correlation",
                                         "duplicate_rows", "stuck_periods", "frozen_hours", "flatlines")},
        "hourly": {"t": [t.strftime("%Y-%m-%d %H") for t in h.index], "silica": rnd(ts, 2), "cls": lab.tolist(),
                   **{f"air{i}": rnd(h[f"air_flow_{i}_mean"], 1) for i in (3, 4, 5)}},
        "daily": {"day": [d.strftime("%Y-%m-%d") for d in daily.index], **{c: rnd(daily[c], 3) for c in daily.columns}},
        "monthly": {"month": monthly.index.tolist(), **{c: rnd(monthly[c], 3) for c in monthly.columns}},
        "conditions": {"label": cond["label"].tolist(), "d": rnd(cond["std_difference"], 3)},
        "response": {"lag": resp.index.tolist(), **{c: rnd(resp[c], 3) for c in resp.columns}},
        "preds": {"t": [t.strftime("%Y-%m-%d %H") for t in preds.index], "y": rnd(preds["y"], 2),
                  "persistence": rnd(preds["persistence"], 2), "shrunk": rnd(preds["shrunk"], 2),
                  "lo": rnd(preds["lo"], 2), "hi": rnd(preds["hi"], 2), "p": rnd(preds["p_offspec"], 3)},
        "scores": {m: {k: rnd(v, 6) for k, v in sc.loc[m].items()} for m in sc.index},
        "sweep": {"delay": sweep.index.tolist(), **{c: rnd(sweep[c], 4) for c in sweep.columns}},
        "importance": {"group": imp.index.tolist(), "mae_increase": rnd(imp["mae_increase"], 4)},
        "learning": {"weeks": lc.index.tolist(), **{c: rnd(lc[c], 4) for c in lc.columns}},
        "training": {"iteration": tc.index.tolist(), **{c: rnd(tc[c], 4) for c in tc.columns}},
        "leakage": {"test": leak["test"].tolist(), "r2": rnd(leak["r2"], 6), "mae": rnd(leak["mae"], 6)},
        "warning": warn,
        "delay": args.delay,
    }
    (out / "plant_dashboard.json").write_text(to_json(dash, compact=True))
    s = results["soft_sensor"]
    print(f"Measured laboratory hours: {audit['lab_classes']['measured']:,} of {audit['hours_with_data']:,} hours with data")
    print(f"Off-specification (silica above {kpi['off_spec_limit_pct']}%): {kpi['off_spec_pct']:.1f}% of measured hours")
    print(f"Soft sensor, {args.delay} h turnaround: MAE {s['mae']['shrunk']:.3f} against {s['mae']['persistence']:.3f} for the latest result "
          f"({s['gain_vs_persistence_pct']:.1f}% lower); process sensors only {s['mae']['gb_process']:.3f}; 80% interval coverage {s['coverage_80']:.1%}")


if __name__ == "__main__":
    main()
