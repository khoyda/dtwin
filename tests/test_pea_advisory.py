"""Tests for the yellow-pea advisory layer (legume logic)."""

from datetime import date

import pytest

from canola_dt.advisory import (
    PeaAdvisoryEngine,
    PeaAgronomyParameters,
    PeaFieldState,
    PeaGrowthStage,
    PeaPrecedingCrop,
    PeaType,
    pea_seeding_rate,
)
from canola_dt.advisory.pea_engine import _fixation
from canola_dt.data.ingest import synthetic_weather


def test_pea_state_json_round_trip():
    s = PeaFieldState(field_id="P1", seeding_date=date(2024, 5, 6),
                      preceding_crop=PeaPrecedingCrop.CEREAL, pea_type=PeaType.YELLOW)
    s.ingest_sensor_reading(50, 24.0, 8.0, 80, 28.0)
    r = PeaFieldState.from_dict(s.to_dict())
    assert r.preceding_crop == PeaPrecedingCrop.CEREAL
    assert r.day_of_season == 50


def test_high_n_suppresses_fixation_alert_and_credit():
    e = PeaAdvisoryEngine()
    p = e.params
    # Starter N -> full fixation; high N -> none, with a CRITICAL alert.
    lo = PeaFieldState(n_applied_kg_per_ha=12)
    hi = PeaFieldState(n_applied_kg_per_ha=80, plant_population_per_m2=80)
    assert _fixation(lo, p) > _fixation(hi, p)
    assert _fixation(hi, p) == 0.0
    hi.day_of_season = 3
    alerts, _ = e.step(hi)
    assert any(a.category == "NitrogenSuppressesFixation" and a.severity.value == "CRITICAL"
               for a in alerts)


def test_pea_on_pea_rotation_critical():
    e = PeaAdvisoryEngine()
    s = PeaFieldState(preceding_crop=PeaPrecedingCrop.PEA, years_since_last_pulse=1,
                      plant_population_per_m2=80)
    s.day_of_season = 3
    alerts, _ = e.step(s)
    assert any(a.category == "CropRotation" and a.severity.value == "CRITICAL" for a in alerts)


def test_missing_inoculant_alert():
    e = PeaAdvisoryEngine()
    s = PeaFieldState(inoculant_applied=False, preceding_crop=PeaPrecedingCrop.CEREAL)
    s.day_of_season = 3
    alerts, _ = e.step(s)
    assert any(a.category == "Inoculation" for a in alerts)


def test_heat_abort_alert_at_flowering():
    e = PeaAdvisoryEngine()
    s = PeaFieldState(plant_population_per_m2=80)
    s.day_of_season = 50  # flowering
    s.air_temp_max_c = 28  # above 25 C abort threshold
    alerts, _ = e.step(s)
    assert any(a.category == "HeatStress" for a in alerts)


def test_update_yield_no_n_modifier_and_protein():
    e = PeaAdvisoryEngine()
    s = PeaFieldState(plant_population_per_m2=80, preceding_crop=PeaPrecedingCrop.CEREAL, latitude=50.5)
    weather = synthetic_weather(year=2022, n_days=150, seed=6)
    e.update_yield(s, weather, 50.5)
    assert s.yield_potential_t_ha > 0
    assert 22.0 <= s.estimated_protein_pct <= 25.0
    assert "n_fixed_kg_ha" in s.yield_breakdown        # fixation reported
    assert "nitrogen_mod" not in s.yield_breakdown     # no N modifier (legume)


def test_pea_seeding_rate_large_seed():
    r = pea_seeding_rate(target_plants_per_m2=80, thousand_kernel_weight_g=235, emergence_pct=88)
    assert r["seeds_per_m2_needed"] == pytest.approx(90.9, abs=0.5)
    assert r["seeding_rate_kg_per_ha"] == pytest.approx(213.6, abs=1.0)  # big seed -> high rate
