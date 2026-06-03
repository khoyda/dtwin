"""Network-free tests for the wheat sub-provincial RM-yield loader."""

import pandas as pd
import pytest

from canola_dt import subprovincial as sp
from canola_dt.data.aafc import CANOLA_BU_AC_TO_KG_HA, WHEAT_BU_AC_TO_KG_HA


def test_wheat_bushel_factor():
    # Wheat 60-lb bushel is heavier than canola 50-lb -> larger kg/ha per bu/ac.
    assert WHEAT_BU_AC_TO_KG_HA == pytest.approx(67.25, abs=0.1)
    assert WHEAT_BU_AC_TO_KG_HA > CANOLA_BU_AC_TO_KG_HA


def test_load_rm_crop_yields_selects_column_and_converts(tmp_path):
    # Pre-create the cache file so no download is attempted.
    pd.DataFrame({
        "Year": [2020, 2020, 2021],
        "RM": [1, 2, 1],
        "Canola": [40, None, 35],
        "Spring Wheat": [50, 45, None],
    }).to_csv(tmp_path / "sk_rm_yields.csv", index=False)

    wheat = sp.load_rm_crop_yields(tmp_path, "Spring Wheat", WHEAT_BU_AC_TO_KG_HA)
    # NaN rows dropped; 50 bu/ac * 67.25 ~ 3362 kg/ha.
    assert len(wheat) == 2
    assert set(wheat.columns) == {"rmno", "year", "yield_kg_ha"}
    row = wheat[(wheat["rmno"] == 1) & (wheat["year"] == 2020)].iloc[0]
    assert row["yield_kg_ha"] == pytest.approx(50 * WHEAT_BU_AC_TO_KG_HA, abs=0.1)
