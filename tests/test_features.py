"""Unit tests for feature engineering."""

import pandas as pd

from canola_dt import constants
from canola_dt.features import (
    cumulative_gdd,
    growing_degree_days,
    heat_stress_days,
    season_features,
)


def test_gdd_floors_at_zero_and_caps():
    tmean = pd.Series([0.0, 5.0, 20.0, 40.0])
    gdd = growing_degree_days(tmean, base_temp_c=5.0, cap_temp_c=30.0)
    # Below/at base -> 0; 20 -> 15; 40 capped to 30 -> 25.
    assert list(gdd) == [0.0, 0.0, 15.0, 25.0]


def test_cumulative_gdd_is_monotonic():
    tmean = pd.Series([10.0, 12.0, 8.0])
    cum = cumulative_gdd(tmean)
    assert list(cum) == sorted(cum)


def test_heat_stress_threshold():
    tmax = pd.Series([28.0, 29.5, 31.0])
    stress = heat_stress_days(tmax, threshold_c=constants.HEAT_STRESS_THRESHOLD_C)
    assert list(stress) == [False, False, True]


def test_season_features_keys():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-05-10", periods=5, freq="D"),
            "tmin_c": [8, 9, 10, 7, 11],
            "tmax_c": [22, 24, 31, 20, 30],
            "tmean_c": [15, 16, 20, 13, 20],
            "precip_mm": [0, 5, 0, 0, 12],
        }
    )
    feats = season_features(df)
    assert {"total_gdd", "total_precip_mm", "heat_stress_days", "max_dry_spell"} <= feats.keys()
    assert feats["heat_stress_days"] == 2
    assert feats["total_precip_mm"] == 17.0
