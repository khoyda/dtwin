"""Network-free tests for sub-provincial validation helpers."""

import numpy as np
import pandas as pd

from canola_dt import subprovincial as sp


def test_nearest_rm_picks_closest_centroid():
    centroids = pd.DataFrame({
        "rmno": [100, 200, 300],
        "lat": [50.0, 52.0, 54.0],
        "lon": [-105.0, -107.0, -109.0],
    })
    # A point near the second centroid.
    assert sp.nearest_rm(52.1, -106.9, centroids) == 200
    assert sp.nearest_rm(49.9, -105.1, centroids) == 100


def test_detrended_anomaly_corr_perfect_match():
    # Within each station: sim anomaly equals the detrended RM-yield anomaly -> corr ~1.
    # Anomaly [-100, +100, +100, -100] is orthogonal to year, so OLS recovers the
    # trend exactly and the detrended RM anomaly equals the sim anomaly.
    pairs = pd.DataFrame({
        "station_id": [1, 1, 1, 1],
        "year": [2000, 2001, 2002, 2003],
        "rm_yield": [900.0, 1200.0, 1300.0, 1200.0],   # 1000 + 100*t + anomaly
        "sim_yield": [1400.0, 1600.0, 1600.0, 1400.0],  # 1500 + anomaly
    })
    r = sp._detrended_anomaly_corr(pairs, "rm_yield", "station_id")
    assert r > 0.99


def test_detrended_anomaly_corr_skips_short_groups():
    pairs = pd.DataFrame({
        "station_id": [1, 1],
        "year": [2000, 2001],
        "rm_yield": [1000.0, 1100.0],
        "sim_yield": [1500.0, 1600.0],
    })
    # Only one group of length 2 (< 3) -> not enough data -> NaN.
    assert np.isnan(sp._detrended_anomaly_corr(pairs, "rm_yield", "station_id"))
