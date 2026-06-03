"""Tests for the advisory (decision-support) layer."""

from datetime import date

import pandas as pd
import pytest

from canola_dt.advisory import (
    AgronomyParameters,
    CanolaAdvisoryEngine,
    CanolaFieldState,
    CultivarType,
    GrowthStage,
    PrecedingCrop,
    Species,
    calculate_seeding_rate,
    estimate_n_requirement,
)
from canola_dt.data.ingest import synthetic_weather


def test_field_state_json_round_trip():
    s = CanolaFieldState(field_id="X1", seeding_date=date(2024, 5, 6),
                         preceding_crop=PrecedingCrop.PEAS)
    s.ingest_sensor_reading(45, 14.0, 22.0, 8.0, 60, 30.0)
    restored = CanolaFieldState.from_dict(s.to_dict())
    assert restored.field_id == "X1"
    assert restored.seeding_date == date(2024, 5, 6)
    assert restored.preceding_crop == PrecedingCrop.PEAS
    assert restored.species == Species.B_NAPUS
    assert restored.day_of_season == 45


def test_agronomy_params_round_trip():
    p = AgronomyParameters()
    assert AgronomyParameters.from_dict(p.to_dict()).plant_density_optimal_min == 50


def test_growth_stage_progression_by_day():
    engine = CanolaAdvisoryEngine()
    s = CanolaFieldState()
    s.day_of_season = 2
    engine._update_growth_stage(s)
    assert s.growth_stage == GrowthStage.GS0_GERMINATION
    s.day_of_season = 50
    engine._update_growth_stage(s)
    assert s.growth_stage == GrowthStage.GS5_FLOWERING
    s.day_of_season = 120
    engine._update_growth_stage(s)
    assert s.growth_stage == GrowthStage.GS9_MATURITY


def test_low_density_triggers_critical_alert():
    engine = CanolaAdvisoryEngine()
    s = CanolaFieldState(plant_density_per_m2=15)  # in 10-20 critical band
    s.day_of_season = 20
    alerts, _ = engine.step(s)
    assert any(a.severity.value == "CRITICAL" and a.category == "PlantDensity" for a in alerts)


def test_heat_stress_alert_at_flowering():
    engine = CanolaAdvisoryEngine()
    s = CanolaFieldState(plant_density_per_m2=60)
    s.day_of_season = 50  # flowering window
    s.air_temp_max_c = 31.0  # above 29.5 threshold
    alerts, _ = engine.step(s)
    assert any(a.category == "HeatStress" for a in alerts)
    assert s.heat_stress_events_at_flowering == 1


def test_short_rotation_triggers_alert():
    engine = CanolaAdvisoryEngine()
    s = CanolaFieldState(years_since_last_canola=1, plant_density_per_m2=60)
    s.day_of_season = 3
    alerts, _ = engine.step(s)
    assert any(a.category == "CropRotation" and a.severity.value == "CRITICAL" for a in alerts)


def test_management_modifiers_optimal_density_no_penalty():
    engine = CanolaAdvisoryEngine()
    s = CanolaFieldState(plant_density_per_m2=60, preceding_crop=PrecedingCrop.WHEAT,
                         cultivar_type=CultivarType.HYBRID, n_applied_kg_per_ha=150)
    mods = engine._management_modifiers(s)
    assert mods["density"] == 1.0       # 60 is within optimal range
    assert mods["rotation"] == 1.0      # wheat baseline
    assert mods["nitrogen"] == 1.0      # 150 >= hybrid threshold
    assert mods["combined"] == 1.0


def test_management_modifiers_apply_penalties():
    engine = CanolaAdvisoryEngine()
    s = CanolaFieldState(plant_density_per_m2=15,                 # critical band -> 0.65
                         preceding_crop=PrecedingCrop.CANOLA,     # 0.85
                         cultivar_type=CultivarType.HYBRID,
                         n_applied_kg_per_ha=50)                  # below 100 -> 0.88
    mods = engine._management_modifiers(s)
    assert mods["density"] == 0.65
    assert mods["rotation"] == 0.85
    assert mods["nitrogen"] == 0.88
    assert mods["combined"] == pytest.approx(0.65 * 0.85 * 0.88, abs=1e-3)


def test_update_yield_uses_process_model():
    engine = CanolaAdvisoryEngine()  # default (uncalibrated) process model
    s = CanolaFieldState(plant_density_per_m2=60, preceding_crop=PrecedingCrop.WHEAT,
                         n_applied_kg_per_ha=150, latitude=50.5)
    weather = synthetic_weather(year=2022, n_days=150, seed=3)
    engine.update_yield(s, weather, latitude=50.5)
    assert s.yield_potential_t_ha > 0
    # bu/ac and t/ha are consistent (1 bu/ac canola = 56.06 kg/ha).
    assert s.yield_potential_bu_ac == pytest.approx(s.yield_potential_t_ha * 1000 / 56.06, abs=0.2)
    assert s.yield_breakdown["biophysical_kg_ha"] > 0


def test_canola_sulphur_starvation_caps_forecast():
    e = CanolaAdvisoryEngine()
    weather = synthetic_weather(year=2022, n_days=150, seed=1)
    base = dict(plant_density_per_m2=60, preceding_crop=PrecedingCrop.PEAS,
                n_applied_kg_per_ha=150, latitude=50.5)
    adequate = CanolaFieldState(s_applied_kg_per_ha=15, **base)
    e.update_yield(adequate, weather, 50.5)
    starved = CanolaFieldState(s_applied_kg_per_ha=0, soil_available_s_kg_per_ha=2, **base)
    e.update_yield(starved, weather, 50.5)
    assert starved.yield_potential_t_ha < adequate.yield_potential_t_ha
    assert starved.yield_breakdown["limiting_factor"] == "S"


def test_canola_fertility_report_flags_sulphur():
    e = CanolaAdvisoryEngine()
    s = CanolaFieldState(s_applied_kg_per_ha=0, soil_available_s_kg_per_ha=2)
    report = e.fertility_report(s, target_yield_t_ha=2.5)
    assert report["limiting_nutrient"] == "S"   # canola high S demand
    assert any(a.startswith("S:") for a in report["deficiency_alerts"])


def test_seeding_rate_calculator():
    r = calculate_seeding_rate(target_density_per_m2=65, thousand_seed_weight_g=5.5)
    assert r["seeds_per_m2_needed"] == pytest.approx(118.2, abs=0.2)
    assert r["seeding_rate_kg_per_ha"] == pytest.approx(6.5, abs=0.1)


def test_n_requirement_hybrid_adds_extra():
    r = estimate_n_requirement(3.0, CultivarType.HYBRID, soil_n_available_kg_per_ha=40)
    assert r["extra_n_for_hybrid_kg_per_ha"] == 55.0
    # 3 t * 55 kg/t = 165 demand; (165-40) + 55 = 180
    assert r["n_fertiliser_recommended_kg_per_ha"] == pytest.approx(180.0, abs=0.5)
