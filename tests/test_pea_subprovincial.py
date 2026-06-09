"""Network-free test for the pea sub-provincial RM-yield loader."""

import pandas as pd
import pytest

from canola_dt import subprovincial as sp
from canola_dt.data.aafc import PEA_BU_AC_TO_KG_HA, WHEAT_BU_AC_TO_KG_HA


def test_pea_bushel_factor_matches_60lb():
    assert PEA_BU_AC_TO_KG_HA == pytest.approx(67.25, abs=0.1)
    assert PEA_BU_AC_TO_KG_HA == pytest.approx(WHEAT_BU_AC_TO_KG_HA)  # both 60-lb bushel


def test_load_rm_pea_yields_selects_column(tmp_path):
    pd.DataFrame({"Year": [2020, 2021], "RM": [1, 1],
                  "Canola": [40, 35], "Peas": [38, None]}).to_csv(tmp_path / "sk_rm_yields.csv", index=False)
    peas = sp.load_rm_crop_yields(tmp_path, "Peas", PEA_BU_AC_TO_KG_HA)
    assert len(peas) == 1  # NaN dropped
    assert peas.iloc[0]["yield_kg_ha"] == pytest.approx(38 * PEA_BU_AC_TO_KG_HA, abs=0.1)
