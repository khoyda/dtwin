"""Network-free unit tests for the data-source helpers."""

import pandas as pd
import pytest

from canola_dt.data.aafc import CANOLA_BU_AC_TO_KG_HA, _to_kg_ha, load_region_yield
from canola_dt.data.eccc import _find_col


def test_find_col_matches_degree_symbol_columns():
    # ECCC encodes the degree symbol inconsistently; match on stable substring.
    cols = ["Date/Time", "Max Temp (\xb0C)", "Min Temp (�C)", "Total Precip (mm)"]
    assert _find_col(cols, "Max Temp") == "Max Temp (\xb0C)"
    assert _find_col(cols, "Total Precip") == "Total Precip (mm)"


def test_find_col_raises_when_absent():
    with pytest.raises(KeyError):
        _find_col(["a", "b"], "Mean Temp")


def test_canola_unit_conversion():
    assert _to_kg_ha(2000, "kg/ha") == 2000.0
    # 40 bu/ac canola -> ~2242 kg/ha
    assert _to_kg_ha(40, "bu/ac") == pytest.approx(40 * CANOLA_BU_AC_TO_KG_HA)
    assert _to_kg_ha(40, "bu/ac") == pytest.approx(2242.5, abs=1.0)
    with pytest.raises(ValueError):
        _to_kg_ha(1, "tonnes/ha")


def test_load_region_yield_normalizes_units(tmp_path):
    csv = tmp_path / "aafc.csv"
    pd.DataFrame(
        {
            "region": ["RM A", "RM B"],
            "province": ["Saskatchewan", "Manitoba"],
            "year": [2020, 2020],
            "yield": [2200, 40],
            "unit": ["kg/ha", "bu/ac"],
        }
    ).to_csv(csv, index=False)

    out = load_region_yield(csv)
    assert list(out.columns) == ["region", "province", "year", "yield_kg_ha"]
    assert out.loc[out["region"] == "RM A", "yield_kg_ha"].iloc[0] == 2200.0
    assert out.loc[out["region"] == "RM B", "yield_kg_ha"].iloc[0] == pytest.approx(2242.5, abs=1.0)
