"""Network-free tests for the calibration helpers."""

import numpy as np
import pandas as pd
import pytest

from canola_dt import calibration as cal


def test_technology_detrend_removes_linear_trend():
    # Yields rising exactly 100 kg/ha/yr -> adjusted to ref year should be flat.
    obs = pd.DataFrame({
        "province": ["SK"] * 3,
        "year": [2000, 2001, 2002],
        "yield_kg_ha": [1000.0, 1100.0, 1200.0],
    })
    out = cal.technology_detrend(obs, ref_year=2002)
    assert out["slope"].iloc[0] == pytest.approx(100.0)
    # Each year expressed at 2002 genetics equals the 2002 yield.
    assert np.allclose(out["adjusted"], 1200.0)


def test_calibration_metrics_perfect_anomaly_match():
    # sim anomaly identical to observed detrended anomaly -> corr 1, anomaly_rmse 0.
    merged = pd.DataFrame({
        "province": ["SK", "SK", "SK"],
        "trend": [2000.0, 2000.0, 2000.0],
        "yield_kg_ha": [1800.0, 2000.0, 2200.0],   # obs anomalies: -200, 0, +200
        "adjusted": [1800.0, 2000.0, 2200.0],
        "sim_yield": [1300.0, 1500.0, 1700.0],      # sim anomalies: -200, 0, +200
    })
    m = cal.calibration_metrics(merged)
    assert m["anomaly_corr"] == 1.0
    assert m["anomaly_rmse"] < 1e-9
    assert m["bias"] < 0  # sim mean below adjusted mean (level handled by offsets)


def test_province_offsets_and_corrected_zero_after_offset():
    merged = pd.DataFrame({
        "province": ["SK", "SK", "MB", "MB"],
        "trend": [2000.0, 2000.0, 2100.0, 2100.0],
        "yield_kg_ha": [1900.0, 2100.0, 2000.0, 2200.0],
        "adjusted": [1900.0, 2100.0, 2000.0, 2200.0],
        "sim_yield": [1400.0, 1600.0, 1700.0, 1900.0],  # each province 500/400 below
    })
    offsets = cal.province_offsets(merged)
    assert offsets["SK"] == 500.0
    assert offsets["MB"] == 300.0
    # Here each province's sim spread matches its adjusted spread exactly, so once
    # the per-province offset is applied the absolute error is zero.
    corrected = cal.corrected_metrics(merged, offsets)
    assert corrected["mae"] == pytest.approx(0.0)
