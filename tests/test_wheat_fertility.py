"""Tests for wheat fertility coefficients and the nutrient-limited yield forecast."""

import pytest

from canola_dt import fertility as fert
from canola_dt.advisory import WheatAdvisoryEngine, WheatFieldState, WheatPrecedingCrop
from canola_dt.data.ingest import synthetic_weather


def test_wheat_nutrient_parameters_distinct_from_canola():
    wp = fert.wheat_nutrient_parameters()
    cp = fert.canola_nutrient_parameters()
    assert wp.uptake_kg_per_t["N"] == 29.0
    assert wp.uptake_kg_per_t != cp.uptake_kg_per_t
    # Mobility strategy: N/S fed (uptake), P/K maintained (removal).
    assert wp.strategy["N"] == "uptake" and wp.strategy["P2O5"] == "removal"


def test_wheat_fertilizer_recommendation():
    wp = fert.wheat_nutrient_parameters()
    rec = fert.fertilizer_recommendation(4.0, soil_supply={"N": 40, "P2O5": 20, "K2O": 300, "S": 6}, params=wp)
    # N uptake-deficit: 29*4 - 40 = 76
    assert rec["N"]["recommended_kg_ha"] == pytest.approx(76.0, abs=0.5)
    # P2O5 removal-replacement: 10*4 - 20 = 20
    assert rec["P2O5"]["recommended_kg_ha"] == pytest.approx(20.0, abs=0.5)
    # K2O removal 5.5*4=22 < 300 soil -> 0
    assert rec["K2O"]["recommended_kg_ha"] == 0.0


def test_wheat_nutrient_limited_yield_limiting():
    wp = fert.wheat_nutrient_parameters()
    nl = fert.nutrient_limited_yield(applied={"N": 20, "P2O5": 40, "K2O": 0, "S": 15},
                                     soil_supply={"N": 20, "P2O5": 25, "K2O": 300, "S": 8}, params=wp)
    # N: (20+20)/29 = 1.38 t is the minimum -> N limits.
    assert nl.limiting_nutrient == "N"
    assert nl.yield_t_ha == pytest.approx(1.38, abs=0.05)


def test_sulphur_starvation_caps_forecast():
    e = WheatAdvisoryEngine()
    weather = synthetic_weather(year=2022, n_days=150, seed=4)
    base = dict(plant_population_per_m2=270, preceding_crop=WheatPrecedingCrop.CANOLA,
                n_applied_kg_per_ha=150, latitude=50.5)
    adequate = WheatFieldState(s_applied_kg_per_ha=15, **base)
    e.update_yield(adequate, weather, 50.5)
    starved = WheatFieldState(s_applied_kg_per_ha=0, soil_available_s_kg_per_ha=2, **base)
    e.update_yield(starved, weather, 50.5)
    assert starved.yield_potential_t_ha < adequate.yield_potential_t_ha
    assert starved.yield_breakdown["limiting_factor"] == "S"
    assert adequate.yield_breakdown["limiting_factor"] == "water/weather"


def test_fertility_report_flags_nitrogen():
    e = WheatAdvisoryEngine()
    s = WheatFieldState(n_applied_kg_per_ha=20, soil_available_n_kg_per_ha=20)
    report = e.fertility_report(s, target_yield_t_ha=4.0)
    assert report["limiting_nutrient"] == "N"
    assert any(a.startswith("N:") for a in report["deficiency_alerts"])
    assert report["recommendation_kg_ha"]["N"] > 0
