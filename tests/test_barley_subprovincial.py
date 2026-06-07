"""Network-free tests for the barley sub-provincial RM-yield loader."""

import pandas as pd
import pytest

from canola_dt import subprovincial as sp
from canola_dt.data.aafc import BARLEY_BU_AC_TO_KG_HA, WHEAT_BU_AC_TO_KG_HA


def test_barley_bushel_factor():
    # Barley 48-lb bushel is lighter than wheat 60-lb -> smaller kg/ha per bu/ac.
    assert BARLEY_BU_AC_TO_KG_HA == pytest.approx(53.80, abs=0.1)
    assert BARLEY_BU_AC_TO_KG_HA < WHEAT_BU_AC_TO_KG_HA


def test_load_rm_barley_yields_selects_column_and_converts(tmp_path):
    pd.DataFrame({
        "Year": [2020, 2020, 2021],
        "RM": [1, 2, 1],
        "Canola": [40, None, 35],
        "Barley": [70, 60, None],
    }).to_csv(tmp_path / "sk_rm_yields.csv", index=False)

    barley = sp.load_rm_crop_yields(tmp_path, "Barley", BARLEY_BU_AC_TO_KG_HA)
    assert len(barley) == 2  # NaN dropped
    row = barley[(barley["rmno"] == 1) & (barley["year"] == 2020)].iloc[0]
    assert row["yield_kg_ha"] == pytest.approx(70 * BARLEY_BU_AC_TO_KG_HA, abs=0.1)
