"""Network-free tests for the NASA POWER gridded-weather parser."""

import pandas as pd

from canola_dt.data import nasapower


def test_parse_power_json_schema_and_fill():
    data = {"properties": {"parameter": {
        "T2M": {"20220501": 15.0, "20220502": 16.0},
        "T2M_MAX": {"20220501": 22.0, "20220502": -999.0},   # missing temp -> NA
        "T2M_MIN": {"20220501": 8.0, "20220502": 9.0},
        "PRECTOTCORR": {"20220501": 5.0, "20220502": -999.0},  # missing precip -> 0
    }}}
    df = nasapower.parse_power_json(data)
    assert list(df.columns) == ["date", "tmin_c", "tmax_c", "tmean_c", "precip_mm"]
    assert len(df) == 2
    assert df["tmax_c"].iloc[0] == 22.0
    assert pd.isna(df["tmax_c"].iloc[1])        # -999 temp -> NA
    assert df["precip_mm"].iloc[1] == 0.0       # -999 precip -> 0
    assert df["date"].iloc[0] == pd.Timestamp("2022-05-01")
