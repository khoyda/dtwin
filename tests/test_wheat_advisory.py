"""Tests for the spring-wheat advisory layer."""

from datetime import date

import pytest

from canola_dt.advisory import (
    WheatAdvisoryEngine,
    WheatAgronomyParameters,
    WheatClass,
    WheatFieldState,
    WheatGrowthStage,
    WheatPrecedingCrop,
    wheat_n_requirement,
    wheat_seeding_rate,
)
from canola_dt.data.aafc import WHEAT_BU_AC_TO_KG_HA
from canola_dt.data.ingest import synthetic_weather


def test_wheat_state_json_round_trip():
    s = WheatFieldState(field_id="W1", seeding_date=date(2024, 5, 6),
                        preceding_crop=WheatPrecedingCrop.PULSE, wheat_class=WheatClass.CPSR)
    s.ingest_sensor_reading(66, 24.0, 10.0, 260, 30.0, relative_humidity_pct=75)
    r = WheatFieldState.from_dict(s.to_dict())
    assert r.field_id == "W1"
    assert r.preceding_crop == WheatPrecedingCrop.PULSE
    assert r.wheat_class == WheatClass.CPSR
    assert r.day_of_season == 66


def test_agronomy_params_round_trip():
    p = WheatAgronomyParameters()
    assert WheatAgronomyParameters.from_dict(p.to_dict()).target_population_per_m2_min == 247


def test_growth_stage_progression():
    e = WheatAdvisoryEngine()
    s = WheatFieldState()
    for day, stage in [(3, WheatGrowthStage.GERMINATION), (30, WheatGrowthStage.JOINTING),
                       (45, WheatGrowthStage.FLAG_LEAF), (53, WheatGrowthStage.BOOT),
                       (66, WheatGrowthStage.ANTHESIS), (105, WheatGrowthStage.MATURITY)]:
        s.day_of_season = day
        e._update_growth_stage(s)
        assert s.growth_stage == stage


def test_fhb_alert_at_anthesis_favourable():
    e = WheatAdvisoryEngine()
    s = WheatFieldState(plant_population_per_m2=270)
    s.day_of_season = 66            # anthesis
    s.air_temp_max_c = 23           # within 20-25
    s.relative_humidity_pct = 80    # >= 70
    alerts, _ = e.step(s)
    fhb = [a for a in alerts if a.category == "FusariumHeadBlight"]
    assert fhb and fhb[0].severity.value == "CRITICAL"
    assert s.fhb_risk_events == 1


def test_midge_alert_in_window_above_threshold():
    e = WheatAdvisoryEngine()
    s = WheatFieldState(plant_population_per_m2=270)
    s.day_of_season = 53            # boot (window open)
    s.may_precipitation_mm = 40     # above emergence requirement
    s.midge_per_head = 0.3          # above ~0.22 yield threshold
    alerts, _ = e.step(s)
    assert any(a.category == "WheatMidge" and a.severity.value == "CRITICAL" for a in alerts)


def test_wheat_on_wheat_rotation_critical():
    e = WheatAdvisoryEngine()
    s = WheatFieldState(preceding_crop=WheatPrecedingCrop.WHEAT, years_since_last_wheat=1,
                        plant_population_per_m2=270)
    s.day_of_season = 3
    alerts, _ = e.step(s)
    assert any(a.category == "CropRotation" and a.severity.value == "CRITICAL" for a in alerts)


def test_management_modifiers_wheat_on_wheat_penalty():
    e = WheatAdvisoryEngine()
    s = WheatFieldState(plant_population_per_m2=270, preceding_crop=WheatPrecedingCrop.WHEAT,
                        n_applied_kg_per_ha=120)
    mods = e._management_modifiers(s)
    assert mods["population"] == 1.0
    assert mods["rotation"] == 0.90      # wheat-on-wheat
    assert mods["nitrogen"] == 1.0
    assert mods["combined"] == pytest.approx(0.90, abs=1e-3)


def test_update_yield_and_protein():
    e = WheatAdvisoryEngine()  # default (uncalibrated) wheat model
    s = WheatFieldState(plant_population_per_m2=270, preceding_crop=WheatPrecedingCrop.CANOLA,
                        n_applied_kg_per_ha=157, latitude=50.5)
    weather = synthetic_weather(year=2022, n_days=150, seed=4)
    e.update_yield(s, weather, latitude=50.5)
    assert s.yield_potential_t_ha > 0
    assert s.yield_potential_bu_ac == pytest.approx(
        s.yield_potential_t_ha * 1000 / WHEAT_BU_AC_TO_KG_HA, abs=0.2)
    # At the yield-maximizing N rate, protein sits at the target.
    assert s.estimated_protein_pct == pytest.approx(13.5, abs=0.2)


def test_calculators():
    sr = wheat_seeding_rate(target_plants_per_m2=275, kernel_weight_g=37, survival_pct=85)
    assert sr["seeds_per_m2_needed"] == pytest.approx(323.5, abs=0.5)
    assert 80 < sr["seeding_rate_lb_per_ac"] < 130  # within CWRS spec range
    n = wheat_n_requirement(4.0, 13.5, soil_n_kg_per_ha=40)
    assert n["n_recommended_kg_per_ha"] == pytest.approx(102.0, abs=1.0)
