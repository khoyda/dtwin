"""Tests for the scenario forecasting layer (network-free via synthetic weather)."""

import pytest

from canola_dt.scenario import Scenario, run_scenario


def test_from_dict_aliases_and_unknown_keys():
    sc = Scenario.from_dict({"crop": "wheat", "preceding": "wheat", "plants": 250})
    assert sc.preceding_crop == "wheat"     # alias preceding -> preceding_crop
    assert sc.plants_per_m2 == 250          # alias plants -> plants_per_m2
    with pytest.raises(ValueError):
        Scenario.from_dict({"crop": "wheat", "nope": 1})


@pytest.mark.parametrize("crop", ["wheat", "canola"])
def test_run_scenario_synthetic(crop):
    r = run_scenario(Scenario(crop=crop, weather="synthetic", analog_year=2022, n=120))
    assert r["crop"] == crop
    assert r["yield_t_ha"] > 0
    assert r["days_to_maturity"] is not None
    assert r["limiting_factor"] in {"water/weather", "N", "P2O5", "K2O", "S"}
    assert (r["protein_pct"] is not None) == (crop == "wheat")


def test_nitrogen_lifts_protein_not_yield_in_dry_season():
    low = run_scenario(Scenario(crop="wheat", weather="synthetic", n=70, preceding_crop="canola"))
    high = run_scenario(Scenario(crop="wheat", weather="synthetic", n=140, preceding_crop="canola"))
    # Synthetic season is water-limited: extra N raises protein, not yield.
    assert high["protein_pct"] > low["protein_pct"]
    assert high["yield_t_ha"] == pytest.approx(low["yield_t_ha"], abs=0.05)


def test_sulphur_starvation_caps_scenario_yield():
    ok = run_scenario(Scenario(crop="canola", weather="synthetic", s=15, preceding_crop="peas"))
    low = run_scenario(Scenario(crop="canola", weather="synthetic", s=0, soil_s=2, preceding_crop="peas"))
    assert low["yield_t_ha"] < ok["yield_t_ha"]
    assert low["limiting_factor"] == "S"
