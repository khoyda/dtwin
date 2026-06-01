"""Behavioural tests for the APSIM-style canola process model.

These assert that the model *responds correctly* to drivers (stress reduces yield,
phenology is ordered, outputs are physically plausible) rather than pinning exact
values — the absolute magnitude awaits calibration against Prairie yield data.
"""

import numpy as np
import pandas as pd
import pytest

from canola_dt.constants import GrowthStage
from canola_dt.simulation.process_model import CanolaCropModel, CanolaParameters

LAT = 50.5


def _constant_season(tmean, tmax, tmin, precip_every, precip_mm, n=150):
    """Deterministic season: constant temps, rain every ``precip_every`` days."""
    dates = pd.date_range("2022-05-01", periods=n, freq="D")
    precip = np.where(np.arange(n) % precip_every == 0, precip_mm, 0.0)
    return pd.DataFrame(
        {
            "date": dates,
            "tmean_c": np.full(n, tmean, float),
            "tmax_c": np.full(n, tmax, float),
            "tmin_c": np.full(n, tmin, float),
            "precip_mm": precip,
        }
    )


@pytest.fixture
def model():
    return CanolaCropModel(CanolaParameters())


def test_favourable_season_runs_to_maturity_with_plausible_yield(model):
    res = model.run(_constant_season(18, 25, 11, 3, 6.0), LAT)
    s = res.summary
    assert s["reached_maturity"] is True
    assert 800 < s["yield_kg_ha"] < 4000        # plausible canola range
    assert 2.5 < s["max_lai"] <= 5.5
    assert s["total_biomass_g_m2"] > 300
    assert 0.0 < s["harvest_index"] <= 0.30


def test_phenology_is_ordered(model):
    res = model.run(_constant_season(18, 25, 11, 3, 6.0), LAT)
    daily = res.daily
    # Thermal time is non-decreasing.
    assert list(daily["tt_cum"]) == sorted(daily["tt_cum"])
    assert s_to(res, GrowthStage.FLOWERING) < s_to(res, GrowthStage.MATURITY)
    # Stages appear in canonical order.
    seen = daily["stage"].drop_duplicates().tolist()
    assert seen == sorted(seen)


def test_heat_and_drought_reduce_yield(model):
    favourable = model.run(_constant_season(18, 25, 11, 3, 6.0), LAT).summary
    hot_dry = model.run(_constant_season(25, 34, 16, 10, 5.0), LAT).summary
    assert favourable["yield_kg_ha"] > hot_dry["yield_kg_ha"]
    # Hot season accrues flowering heat-stress days; favourable mild one does not.
    assert hot_dry["flowering_heat_days"] > favourable["flowering_heat_days"]


def test_drought_increases_water_stress(model):
    wet = model.run(_constant_season(18, 25, 11, 2, 10.0), LAT).summary
    dry = model.run(_constant_season(18, 25, 11, 15, 4.0), LAT).summary
    assert dry["mean_flowering_water_stress"] > wet["mean_flowering_water_stress"]
    assert wet["yield_kg_ha"] > dry["yield_kg_ha"]


def s_to(result, stage: GrowthStage) -> int:
    """Days from start to first day at/after ``stage`` (helper for tests)."""
    daily = result.daily
    start = daily["date"].iloc[0]
    hit = daily[daily["stage"] >= int(stage)]
    return int((hit["date"].iloc[0] - start).days)
