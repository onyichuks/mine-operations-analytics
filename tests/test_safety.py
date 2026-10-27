"""Unit tests for injury rates, compliance summaries and the injury rate model, on small synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mineops import compliance, risk, safety  # noqa: E402


def make_prod(rows):
    return pd.DataFrame(rows, columns=["MINE_ID", "year", "subunit_cd", "hours", "employees"])


def make_acc(rows):
    a = pd.DataFrame(rows, columns=["MINE_ID", "year", "subunit_cd", "degree", "contractor", "mine_exper", "days_lost", "days_restrict"])
    a["CLASSIFICATION"] = "HANDLING OF MATERIALS"
    return a


def test_injury_flags_follow_degree_codes():
    a = make_acc([("1", 2020, "03", d, False, 2.0, 0, 0) for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    f = safety.flag_injuries(a)
    assert f["all_injury"].sum() == 6
    assert f["nfdl"].sum() == 4
    assert f["fatal"].sum() == 1


def test_annual_rates_use_operator_injuries_and_hours():
    prod = make_prod([("1", 2020, "03", 400_000, 200), ("2", 2020, "30", 600_000, 300),
                      ("1", 2021, "03", 500_000, 250), ("1", 2022, "03", 100_000, 50)])
    acc = make_acc([("1", 2020, "03", 3, False, 0.5, 10, 0),     # operator, lost time
                    ("2", 2020, "30", 6, False, 4.0, 0, 0),      # operator, medical only
                    ("1", 2020, "03", 3, True, 0.1, 5, 0),       # contractor: counted, not in the rate
                    ("1", 2021, "03", 1, False, 9.0, 0, 0),      # fatal
                    ("1", 2022, "03", 3, False, 1.0, 0, 0)])
    assert safety.last_full_year(prod, acc) == 2021
    t = safety.annual(prod, acc, first=2020)
    assert list(t.index) == [2020, 2021]
    assert t.loc[2020, "all_injury_rate"] == pytest.approx(2 / 1_000_000 * 200_000)
    assert t.loc[2020, "nfdl_rate"] == pytest.approx(1 / 1_000_000 * 200_000)
    assert t.loc[2020, "contractor_all_injury"] == 1
    assert t.loc[2021, "operator_fatal"] == 1
    sub = safety.by_subunit(prod, acc, range(2020, 2021))
    assert sub.loc["30", "all_injury_rate"] == pytest.approx(1 / 600_000 * 200_000)
    assert sub.loc["03", "subunit"] == "Open pit"


def test_standard_codes_and_descriptions():
    s = compliance.standard_code(pd.Series(["56.14100(b)", " 57.3360 ", None]))
    assert list(s) == ["56.14100", "57.3360", ""]
    assert compliance.describe("57.14100") == compliance.describe("56.14100") != ""
    assert compliance.describe("47.41") == "Hazard communication: container labels"
    assert compliance.describe("99.999") == ""


def test_activity_groups():
    g = compliance.activity_group(pd.Series(["Regular Safety and Health Inspection", "Verbal Hazard Complaint Inspections",
                                             "Compliance Follow-up Inspection", "Non-Fatal Accident Investigation", "Spot Inspection", "Part 50 Audit"]))
    assert list(g) == ["Regular inspection", "Hazard complaint", "Compliance follow-up", "Accident investigation", "Spot inspection", "Other"]


def test_poisson_deviance():
    y = np.array([0.0, 1.0, 3.0])
    assert risk.poisson_deviance(y, y + 1e-12) == pytest.approx(0.0, abs=1e-6)
    assert risk.poisson_deviance(y, np.array([1.0, 1.0, 1.0])) > 0


def synthetic_mines(n_mines=60, years=range(2005, 2026), seed=0):
    rng = np.random.default_rng(seed)
    prod, acc, vio, ins = [], [], [], []
    for m in range(n_mines):
        mid = f"{m:07d}"
        rate = rng.gamma(4, 0.25)                       # each mine has its own true rate per 200,000 hours
        size = rng.uniform(20_000, 600_000)
        for y in years:
            hrs = size * rng.uniform(0.8, 1.2)
            prod.append((mid, y, "03", hrs * 0.7, hrs / 2000 * 0.7))
            prod.append((mid, y, "30", hrs * 0.3, hrs / 2000 * 0.3))
            for _ in range(rng.poisson(rate * hrs / 200_000)):
                acc.append((mid, y, "03", 3, False, rng.uniform(0, 10), 10, 0))
            for _ in range(rng.poisson(5)):
                vio.append({"MINE_ID": mid, "year": y, "ss": rng.random() < 0.2, "order": False, "high_negligence": False,
                            "penalty": 100.0, "contractor": False})
            ins.append({"MINE_ID": mid, "year": y, "insp_hours": 40.0, "ACTIVITY": "Regular Safety and Health Inspection", "EVENT_NO": f"{mid}{y}"})
    return make_prod(prod), make_acc(acc), pd.DataFrame(vio), pd.DataFrame(ins)


def test_panel_targets_next_year_and_filters_small_mines():
    prod, acc, vio, ins = synthetic_mines(n_mines=8, years=range(2018, 2023))
    df = risk.panel(prod, acc, vio, ins, min_hours=10_000, last_year=2022)
    assert df["target_year"].max() == 2022
    assert (df["target_year"] == df["year"] + 1).all()
    row = df.iloc[0]
    nxt = acc[(acc["MINE_ID"] == row["MINE_ID"]) & (acc["year"] == row["year"] + 1)]
    assert row["next_nfdl"] == len(nxt)
    hrs_next = prod[(prod["MINE_ID"] == row["MINE_ID"]) & (prod["year"] == row["year"] + 1)]["hours"].sum()
    assert row["exposure"] == pytest.approx(hrs_next / 200_000)
    assert (df["hours"] >= 10_000).all()
    assert df["share_mill"].between(0.29, 0.31).all()


def test_rate_model_learns_mine_rates():
    prod, acc, vio, ins = synthetic_mines()
    df = risk.panel(prod, acc, vio, ins, last_year=2025)
    r = risk.evaluate(df, test_from=2020)
    st = risk.score_table(r)
    assert st.loc["Industry rate", "deviance_skill_pct"] == pytest.approx(0.0)
    # each synthetic mine has a stable true rate, so its own history must beat the industry rate
    assert st.loc["Own recent rate, shrunk", "deviance_skill_pct"] > 5
    assert st.loc["Own recent rate, shrunk", "top20_rate_ratio"] > 1.2
    bands = risk.risk_bands(r, col="Own recent rate, shrunk")
    assert bands["observed_rate"].iloc[-1] > bands["observed_rate"].iloc[0]
    g = risk.bootstrap_skill_gain(r, "Own recent rate, shrunk", "Industry rate", n=50)
    assert g["low"] <= g["gain_points"] <= g["high"]


def test_top_band_shares():
    r = pd.DataFrame({"target_year": [2020] * 5, "next_nfdl": [5, 0, 0, 0, 0], "next_hours": [100.0] * 5,
                      "pred": [9, 1, 2, 3, 4]})
    tb = risk.top_band(r, "pred", frac=0.2)
    assert tb["injury_share_pct"] == pytest.approx(100.0)
    assert tb["hours_share_pct"] == pytest.approx(20.0)
    assert tb["rate_ratio"] == pytest.approx(5.0)
