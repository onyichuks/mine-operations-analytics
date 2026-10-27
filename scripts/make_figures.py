"""Static figures for the README and report, drawn from the analysis outputs.

Usage:  python scripts/make_figures.py   (after run_plant.py and run_safety.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
S1, S2, S3, S4, INK, INK2, LINE, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0f1a24", "#475665", "#d6dee5", "#b8c3cc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight", "legend.frameon": False})
MONTHS = {"2017-03": "Mar", "2017-04": "Apr", "2017-05": "May", "2017-06": "Jun", "2017-07": "Jul", "2017-08": "Aug", "2017-09": "Sep"}


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, color=INK)


def dates(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def main() -> None:
    FIG.mkdir(exist_ok=True)
    P = json.loads((ROOT / "outputs" / "plant_dashboard.json").read_text())
    S = json.loads((ROOT / "outputs" / "safety_dashboard.json").read_text())

    # 1. which silica values are real measurements
    H = P["hourly"]
    t = pd.to_datetime([x + ":00" for x in H["t"]])
    sil = np.array([np.nan if v is None else v for v in H["silica"]], dtype=float)
    cls = np.array(H["cls"])
    lc = P["audit"]["lab_classes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.4, 1]}, layout="constrained")
    win = (t >= "2017-04-17") & (t < "2017-04-24")
    a1.plot(t[win], sil[win], color=MUTED, lw=1)
    groups = [("measured", "Measured", S1), ("repeat", "Repeat of previous hour", S2), ("varies within hour", "Varies within hour", S3), ("interpolated", "Straight-line fill", S4)]
    for k, lab, col in groups:
        m = win & (cls == k)
        a1.scatter(t[m], sil[m], s=16, color=col, label=lab, edgecolors="white", linewidths=0.6, zorder=3)
    a1.set_ylabel("Silica in concentrate (%)")
    a1.legend(loc="upper left", ncol=4, fontsize=8)
    dates(a1)
    title(a1, "One week of hourly silica values, coloured by what each value really is")
    names = [g[1] for g in groups]
    vals = [lc[g[0]] for g in groups]
    a2.barh(names, vals, color=[g[2] for g in groups])
    a2.invert_yaxis()
    for i, v in enumerate(vals):
        a2.text(v + 40, i, f"{v:,}", va="center", fontsize=9, color=INK2)
    a2.set_xlim(0, max(vals) * 1.25)
    a2.set_xlabel("Hours")
    title(a2, f"{100 * lc['measured'] / sum(vals):.0f}% are real measurements")
    fig.savefig(FIG / "fig01_lab_values.png")
    plt.close(fig)

    # 2. stuck sensors
    n = 24 * 35
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for c, col in [(3, S1), (4, S2), (5, S3)]:
        y = np.array([np.nan if v is None else v for v in H[f"air{c}"][:n]], dtype=float)
        ax.plot(t[:n], y, color=col, lw=1.4, label=f"Column {c}")
    ax.set_ylabel("Air flow (plant units)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    dates(ax)
    st = P["audit"]["stuck_periods"]
    hrs = max(s["hours_of_data"] for s in st) if st else 0
    title(ax, f"Column 4 and 5 air flows repeat one reading for {hrs:.0f} hours while column 3 moves (gap: no data)")
    fig.savefig(FIG / "fig02_stuck_sensors.png")
    plt.close(fig)

    # 3. how the usual test overstates accuracy
    L = P["leakage"]
    short = ["Random 20-second rows,\niron in concentrate included", "Random 20-second rows,\niron removed", "Random measured hours,\nprocess sensors only",
             "Next month,\nprocess sensors only", "Next month,\nlatest laboratory result", "Next month,\nsoft sensor",
             "Next month, soft sensor,\nfilled-in hours treated as measured"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cols = [MUTED] * 3 + [S1] * 3 + [MUTED]
    ax.barh(short, L["r2"], color=cols)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    for i, v in enumerate(L["r2"]):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    ax.set_xlabel("R² on the test set")
    ax.set_xlim(min(L["r2"]) - 0.12, 1.02)
    title(ax, "The same data tested seven ways: leaky tests in grey, tests that mirror use in blue")
    fig.savefig(FIG / "fig03_leakage.png")
    plt.close(fig)

    # 4. soft sensor, one test week
    R = P["preds"]
    tp = pd.to_datetime([x + ":00" for x in R["t"]])
    w = (tp >= "2017-08-07") & (tp < "2017-08-14")
    arr = lambda k: np.array([np.nan if v is None else v for v in R[k]], dtype=float)[w]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(tp[w], arr("lo"), arr("hi"), color=S1, alpha=0.15, lw=0, label="80% range")
    ax.step(tp[w], arr("persistence"), where="post", color=S2, lw=1.6, label=f"Latest result available ({P['delay']} hours old or more)")
    ax.plot(tp[w], arr("shrunk"), color=S1, lw=1.8, label="Soft sensor estimate")
    ax.scatter(tp[w], arr("y"), s=10, color=INK, label="Laboratory result for the hour", zorder=3)
    ax.axhline(P["kpi"]["off_spec_limit_pct"], color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("Silica in concentrate (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8)
    dates(ax)
    sc = P["scores"]["all"]
    title(ax, f"Test week in August 2017: soft sensor error {sc['shrunk_mae']:.3f} against {sc['persistence_mae']:.3f} for the latest result (June to September)")
    fig.savefig(FIG / "fig04_soft_sensor.png")
    plt.close(fig)

    # 5. value against laboratory turnaround, and what the model uses
    W = P["sweep"]
    I = P["importance"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1]}, layout="constrained")
    a1.plot(W["delay"], W["persistence_mae"], color=S2, lw=2, marker="o", ms=4, label="Latest laboratory result")
    a1.plot(W["delay"], W["shrunk_lab_only_mae"], color=S3, lw=2, marker="o", ms=4, label="Laboratory history only (shrunk the same way)")
    a1.plot(W["delay"], W["shrunk_mae"], color=S1, lw=2, marker="o", ms=4, label="Soft sensor (history and sensors)")
    a1.plot(W["delay"], W["train_mean_mae"], color=MUTED, lw=1.5, ls=":", label="Training average")
    a1.set_xlabel("Laboratory turnaround (hours)")
    a1.set_ylabel("Mean absolute error")
    a1.set_ylim(0, None)
    a1.legend(loc="lower right", fontsize=8)
    title(a1, "The soft sensor helps more as results get older")
    a2.barh(I["group"], I["mae_increase"], color=S1)
    a2.invert_yaxis()
    a2.set_xlabel("Increase in error when the group is shuffled")
    title(a2, "Laboratory history carries the signal")
    fig.savefig(FIG / "fig05_turnaround_and_importance.png")
    plt.close(fig)

    # 6. operators respond to the laboratory
    Rp = P["response"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, lab, col in [("amina_flow", "Amine flow", S1), ("starch_flow", "Starch flow", S2), ("air_flow_1", "Column 1 air flow", S3), ("level_5", "Column 5 level", S4)]:
        ax.plot(Rp["lag"], Rp[k], color=col, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Silica result from this many hours earlier (negative: later)")
    ax.set_ylabel("Correlation")
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    title(ax, "Amine flow follows the silica result from about three hours earlier")
    fig.savefig(FIG / "fig06_operator_response.png")
    plt.close(fig)

    # 7. gold mine injury rates
    A = S["annual"]
    T = S["subunit_trend"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    a1.plot(A["year"], A["all_injury_rate"], color=S1, lw=2, marker="o", ms=3, label="All injuries")
    a1.plot(A["year"], A["nfdl_rate"], color=S2, lw=2, marker="o", ms=3, label="Lost-time injuries")
    a1.set_ylabel("Per 200,000 employee hours")
    a1.set_ylim(0, None)
    a1.legend(loc="upper right")
    title(a1, "Gold mine operator injury rates")
    for k, col in [("Open pit", S1), ("Underground", S2), ("Mill or preparation plant", S3), ("Surface at underground mine", S4)]:
        a2.plot(T["year"], T[k], color=col, lw=1.8, marker="o", ms=3, label=k)
    a2.set_ylabel("Lost-time injuries per 200,000 hours")
    a2.set_ylim(0, None)
    a2.legend(loc="upper right", fontsize=8)
    title(a2, "Lost-time injury rate by part of the mine")
    fig.savefig(FIG / "fig07_injury_rates.png")
    plt.close(fig)

    # 8. causes of lost-time injuries and of deaths
    C = S["classes"]
    F = S["fatalities"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    a1.barh(C["name"], C["share_pct"], color=[S2 if n == "Powered haulage" else S1 for n in C["name"]])
    a1.invert_yaxis()
    a1.set_xlabel("% of lost-time injuries")
    title(a1, f"Lost-time injuries, {S['recent'][0]} to {S['recent'][1]}")
    a2.barh(F["name"], F["n"], color=[S2 if n == "Powered haulage" else S1 for n in F["name"]])
    a2.invert_yaxis()
    a2.set_xlabel("Deaths")
    title(a2, f"Deaths, 2000 to {S['last']}")
    fig.savefig(FIG / "fig08_injury_causes.png")
    plt.close(fig)

    # 9. inspections and violations
    Cm = S["compliance"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")
    a1.plot(Cm["year"], Cm["violations_per_100_insp_hours"], color=S1, lw=2, marker="o", ms=3, label="All violations")
    a1.plot(Cm["year"], Cm["ss_per_100_insp_hours"], color=S2, lw=2, marker="o", ms=3, label="Significant and substantial")
    a1.set_ylabel("Per 100 inspection hours")
    a1.set_ylim(0, None)
    a1.legend(loc="upper right")
    title(a1, "Violations found per 100 inspection hours")
    a2.bar(Cm["year"], Cm["ss_share_pct"], color=S1)
    a2.set_ylabel("% of violations")
    title(a2, "Share judged significant and substantial")
    fig.savefig(FIG / "fig09_inspections_violations.png")
    plt.close(fig)

    # 10. injury rate model
    M = S["model"]
    B = M["bands"]
    Y = M["by_year"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    x = np.arange(len(B["band"]))
    a1.bar(x, B["observed"], color=S1, label="Observed next year")
    a1.scatter(x, B["predicted"], color=S2, marker="D", s=40, zorder=3, label="Predicted")
    a1.set_xticks(x, B["band"])
    a1.set_ylabel("Lost-time injuries per 200,000 hours")
    a1.legend(loc="upper left")
    title(a1, "Mines ranked by predicted rate (boosting): observed next-year rate")
    for k, col in [("Own recent rate, shrunk", S1), ("Boosting: size, mix and injury history", S2), ("Boosting: all inputs", S3)]:
        a2.plot(Y["year"], Y[k], color=col, lw=2, marker="o", ms=3, label=k.replace("Own recent rate, shrunk", "Own three-year relative rate, shrunk"))
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_ylabel("Skill beyond last year's industry rate (%)")
    a2.legend(loc="lower left", fontsize=8)
    title(a2, "Skill by test year: inspection records add nothing")
    fig.savefig(FIG / "fig10_injury_rate_model.png")
    plt.close(fig)

    # 11. machine learning checks: training curve and learning curve
    Tc = P["training"]
    Lc = P["learning"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")
    a1.plot(Tc["iteration"], Tc["train_mae"], color=S1, lw=2, label="Training months")
    a1.plot(Tc["iteration"], Tc["valid_mae"], color=S2, lw=2, label="Next month (July)")
    a1.set_xlabel("Boosting iterations")
    a1.set_ylabel("Mean absolute error")
    a1.legend(loc="upper right")
    title(a1, "Training curve: validation error bottoms out early, then drifts up slightly")
    a2.plot(Lc["train_hours"], Lc["test_mae"], color=S1, lw=2, marker="o", ms=4, label="Boosting, history and sensors (unshrunk)")
    a2.plot(Lc["train_hours"], Lc["persistence_mae"], color=S2, lw=1.5, ls="--", label="Latest laboratory result")
    a2.set_xlabel("Measured hours used for training")
    a2.set_ylabel("Mean absolute error")
    a2.legend(loc="upper right")
    title(a2, "Learning curve: level with the latest result from about 14 weeks")
    fig.savefig(FIG / "fig11_ml_curves.png")
    plt.close(fig)
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
