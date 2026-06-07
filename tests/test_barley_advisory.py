"""Tests for the spring-barley advisory layer."""

from datetime import date

import pytest

from canola_dt.advisory import (
    BarleyAdvisoryEngine,
    BarleyAgronomyParameters,
    BarleyFieldState,
    BarleyGrowthStage,
    BarleyPrecedingCrop,
    BarleyType,
    barley_seeding_rate,
)
from canola_dt.data.ingest import synthetic_weather


def test_barley_state_json_round_trip():
    s = BarleyFieldState(field_id="B1", barley_type=BarleyType.FEED,
                         seeding_date=date(2024, 5, 6), preceding_crop=BarleyPrecedingCrop.PULSE)
    s.ingest_sensor_reading(40, 22.0, 10.0, 240, 28.0)
    r = BarleyFieldState.from_dict(s.to_dict())
    assert r.barley_type == BarleyType.FEED
    assert r.preceding_crop == BarleyPrecedingCrop.PULSE
    assert r.day_of_season == 40


def test_agronomy_params_round_trip_and_is_malt():
    p = BarleyAgronomyParameters()
    assert p.is_malt() is True
    assert BarleyAgronomyParameters.from_dict(p.to_dict()).malt_protein_max_pct == 12.5


def test_growth_stage_progression():
    e = BarleyAdvisoryEngine()
    s = BarleyFieldState()
    for day, stage in [(3, BarleyGrowthStage.GERMINATION), (30, BarleyGrowthStage.JOINTING),
                       (55, BarleyGrowthStage.HEADING), (62, BarleyGrowthStage.ANTHESIS),
                       (100, BarleyGrowthStage.MATURITY)]:
        s.day_of_season = day
        e._update_growth_stage(s)
        assert s.growth_stage == stage


def test_malt_protein_dilemma():
    """High N raises protein over the malt ceiling -> malt grade fails (feed is fine)."""
    e = BarleyAdvisoryEngine()  # default malt_2row
    weather = synthetic_weather(year=2022, n_days=140, seed=5)
    lo = BarleyFieldState(barley_type=BarleyType.MALT_2ROW, n_applied_kg_per_ha=90, latitude=50.5,
                          preceding_crop=BarleyPrecedingCrop.CANOLA)
    lo.plant_population_per_m2 = 250
    e.update_yield(lo, weather, 50.5)
    hi = BarleyFieldState(barley_type=BarleyType.MALT_2ROW, n_applied_kg_per_ha=160, latitude=50.5,
                          preceding_crop=BarleyPrecedingCrop.CANOLA)
    hi.plant_population_per_m2 = 250
    e.update_yield(hi, weather, 50.5)
    assert hi.estimated_protein_pct > lo.estimated_protein_pct
    assert lo.malt_grade_ok is True      # ~12.2% in band
    assert hi.malt_grade_ok is False     # protein over 12.5% -> rejected

    # Feed barley has no protein ceiling.
    feed = BarleyFieldState(barley_type=BarleyType.FEED, n_applied_kg_per_ha=160, latitude=50.5)
    feed.plant_population_per_m2 = 250
    e_feed = BarleyAdvisoryEngine(BarleyAgronomyParameters(barley_type=BarleyType.FEED))
    e_feed.update_yield(feed, weather, 50.5)
    assert feed.malt_grade_ok is True


def test_net_blotch_alert_at_flag_leaf():
    e = BarleyAdvisoryEngine()
    s = BarleyFieldState()
    s.growth_stage = BarleyGrowthStage.FLAG_LEAF
    s.net_blotch_severity_pct = 10
    assert any(a.category == "NetBlotch" for a in e._check_net_blotch(s))


def test_barley_on_barley_rotation_critical():
    e = BarleyAdvisoryEngine()
    s = BarleyFieldState(preceding_crop=BarleyPrecedingCrop.BARLEY, years_since_last_barley=1)
    s.day_of_season = 3
    alerts, _ = e.step(s)
    assert any(a.category == "CropRotation" and a.severity.value == "CRITICAL" for a in alerts)


def test_barley_seeding_rate():
    r = barley_seeding_rate(target_plants_per_m2=250, thousand_kernel_weight_g=45, survival_pct=90)
    assert r["seeds_per_m2_needed"] == pytest.approx(277.8, abs=0.5)
    assert 1.0 < r["seeding_rate_bu_per_ac"] < 3.0
