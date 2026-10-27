"""Unit tests for the plant data checks, KPIs and soft sensor, on small synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mineops import data, plant_kpis, plant_quality, softsensor  # noqa: E402
from mineops.data import FEED, PROCESS  # noqa: E402


def rows_for_hours(silica_by_hour: list, start="2017-03-10 00:00", within=None, rng=None) -> pd.DataFrame:
    """Twenty-second rows; each hour's silica is constant unless listed in `within` (then it ramps)."""
    rng = rng or np.random.default_rng(0)
    frames = []
    for i, s in enumerate(silica_by_hour):
        if s is None:
            continue
        t = pd.Timestamp(start) + pd.Timedelta(hours=i)
        f = pd.DataFrame({"time": [t] * plant_quality.ROWS_PER_HOUR})
        for c in PROCESS:
            f[c] = 100 + rng.normal(0, 1, len(f))
        for c in FEED:
            f[c] = 50.0
        f["iron_conc"] = 65.0
        f["silica_conc"] = np.linspace(s, s + 0.5, len(f)) if within and i in within else s
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_decimal_commas_are_converted():
    raw = pd.DataFrame({"date": ["2017-03-10 01:00:00"], **{k: ["1,5"] for k in data.FLOTATION_COLUMNS}})
    out = data.flotation_to_numeric(raw)
    assert out["silica_conc"].iloc[0] == pytest.approx(1.5)
    assert out["time"].iloc[0] == pd.Timestamp("2017-03-10 01:00")
    assert set(data.FLOTATION_COLUMNS.values()) <= set(out.columns)


def test_lab_values_are_classified():
    # hours: measured, measured, repeat, measured, [gap], measured, line, line, measured, ramp within hour
    sil = [1.0, 2.0, 2.0, 3.0, None, 1.0, 1.5, 2.0, 2.5, 4.0]
    df = rows_for_hours(sil, within={9})
    h = plant_quality.hourly(df)
    lab = plant_quality.classify_lab(h)
    assert lab.iloc[2] == "repeat"
    assert lab.iloc[4] == "missing"
    assert lab.iloc[6] == "interpolated" and lab.iloc[7] == "interpolated"
    assert lab.iloc[9] == "varies within hour"
    assert (lab.iloc[[0, 1, 3, 5]] == "measured").all()
    gaps = plant_quality.gaps(h)
    assert gaps == [{"start": h.index[4], "end": h.index[4], "hours": 1}]


def test_stuck_sensor_and_frozen_hours_are_found_and_blanked():
    df = rows_for_hours([1.0 + 0.1 * i for i in range(30)])
    df.loc[: 26 * plant_quality.ROWS_PER_HOUR - 1, "air_flow_4"] = 295.0     # frozen for 26 hours
    stuck = plant_quality.stuck_periods(df)
    assert [p["tag"] for p in stuck] == ["air_flow_4"]
    assert stuck[0]["hours_of_data"] == pytest.approx(26.0)
    h = plant_quality.hourly(df)
    frozen = plant_quality.frozen_hours(h)
    assert not frozen.any()
    h.loc[h.index[28], [f"{t}_std" for t in PROCESS]] = 0.0
    frozen = plant_quality.frozen_hours(h)
    assert frozen.sum() == 1
    hc = plant_quality.clean_hourly(h, stuck, frozen)
    assert hc["air_flow_4_mean"].iloc[:26].isna().all()
    assert hc["air_flow_4_mean"].iloc[27] == pytest.approx(h["air_flow_4_mean"].iloc[27])
    assert hc["amina_flow_mean"].iloc[28] != hc["amina_flow_mean"].iloc[28]   # NaN


def test_feed_update_interval():
    idx = pd.date_range("2017-03-10", periods=40, freq="h")
    h = pd.DataFrame({"iron_feed": np.repeat([50.0, 51.0, 52.0, 53.0, 54.0], 8)}, index=idx)
    assert plant_quality.update_interval_hours(h, "iron_feed") == 8.0


def test_off_spec_kpis_use_measured_hours_only():
    sil = [2.0, 4.0, 4.0, 2.5, 3.5]       # third hour is a repeat and must not count
    df = rows_for_hours(sil)
    h = plant_quality.hourly(df)
    lab = plant_quality.classify_lab(h)
    s = plant_kpis.summary(h, lab)
    assert s["measured_hours"] == 4
    assert s["off_spec_pct"] == pytest.approx(50.0)


def test_lab_history_never_looks_ahead():
    idx = pd.date_range("2017-03-10", periods=12, freq="h")
    h = pd.DataFrame({"silica_conc": np.arange(12, dtype=float) + 1}, index=idx)
    lab = pd.Series("measured", index=idx)
    lab.iloc[5] = "repeat"
    f = softsensor.lab_features(h, lab, delay=2)
    for i in range(len(idx)):
        v = f["lab_last"].iloc[i]
        if not np.isnan(v):
            source = int(v) - 1                 # the hour whose result this is
            assert source <= i - 2              # at least `delay` hours old
            assert source != 5                  # repeats are never used
            assert f["lab_age"].iloc[i] >= 2
    assert np.isnan(f["lab_last"].iloc[1])


def test_conformal_quantile_and_weight():
    r = np.arange(1, 11, dtype=float)            # absolute residuals 1..10
    assert softsensor._conformal_q(r, 0.8) == 9.0   # ceil(11 * 0.8) = 9th smallest
    y = np.array([1.0, 2.0, 3.0])
    last = np.array([0.0, 0.0, 0.0])
    assert softsensor._best_weight(y, last, y * 2) == pytest.approx(0.5)


def synthetic_plant(months=5, seed=1):
    """Hourly frame shaped like plant_quality.hourly output, with silica following an AR(1) process."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-03-01", "2017-09-30 23:00", freq="h")
    n = len(idx)
    s = np.empty(n)
    s[0] = 2.0
    driver = rng.normal(0, 1, n)
    for i in range(1, n):
        s[i] = 2.0 + 0.8 * (s[i - 1] - 2.0) + 0.15 * driver[i] + rng.normal(0, 0.3)
    h = pd.DataFrame(index=idx)
    for t in PROCESS:
        h[f"{t}_mean"] = rng.normal(100, 1, n)
        h[f"{t}_std"] = rng.uniform(0.5, 1.5, n)
    h["amina_flow_mean"] = 100 + driver
    for t in FEED:
        h[t] = 50.0
    h["iron_conc"] = 65.0
    h["silica_conc"] = s
    h["silica_nunique"] = 1
    h["rows"] = 180
    return h


def test_rolling_evaluation_runs_and_is_calibrated():
    h = synthetic_plant()
    lab = pd.Series("measured", index=h.index)
    params = dict(softsensor.GB_PARAMS, max_iter=40)
    preds, models = softsensor.rolling_eval(h, lab, delay=2, params=params, classify=True, keep_models=True)
    assert set(preds["month"]) == {"2017-06", "2017-07", "2017-08", "2017-09"}
    assert preds["weight"].between(0, 1).all()
    sc = softsensor.scores(preds)
    assert 0.7 <= sc.loc["all", "coverage_80"] <= 0.9
    # an AR(1) series is predictable from its own history, so the learned models must beat the training mean
    assert sc.loc["all", "gb_lab_only_mae"] < sc.loc["all", "train_mean_mae"]
    w = softsensor.warning_scores(preds)
    assert 0.5 <= w["model"]["auc"] <= 1.0
    imp = softsensor.grouped_importance(h, lab, 2, None, models, preds, repeats=2)
    assert imp.index[0] == "Laboratory history"
