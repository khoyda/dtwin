"""Behavioural tests for the spring-barley process model."""

import numpy as np
import pandas as pd
import pytest

from canola_dt.simulation.barley_model import BarleyCropModel, BarleyParameters, BarleyStage

LAT = 50.5


def _constant_season(tmean, tmax, tmin, precip_every, precip_mm, n=140):
    dates = pd.date_range("2022-05-01", periods=n, freq="D")
    precip = np.where(np.arange(n) % precip_every == 0, precip_mm, 0.0)
    return pd.DataFrame({
        "date": dates, "tmean_c": np.full(n, tmean, float), "tmax_c": np.full(n, tmax, float),
        "tmin_c": np.full(n, tmin, float), "precip_mm": precip,
    })


@pytest.fixture
def model():
    return BarleyCropModel(BarleyParameters())


def _stage_day(result, stage):
    daily = result.daily
    start = daily["date"].iloc[0]
    hit = daily[daily["stage"] >= int(stage)]
    return int((hit["date"].iloc[0] - start).days)


def test_favourable_season_runs_to_maturity(model):
    s = model.run(_constant_season(17, 23, 9, 3, 6.0), LAT).summary
    assert s["reached_maturity"] is True
    assert 1500 < s["yield_kg_ha"] < 6500
    assert 0.0 < s["harvest_index"] <= 0.48


def test_barley_shorter_season_than_wheat(model):
    # Barley matures faster than wheat under the same favourable season.
    from canola_dt.simulation.wheat_model import WheatCropModel, WheatParameters
    season = _constant_season(17, 23, 9, 3, 6.0)
    b = model.run(season, LAT).summary["days_to_maturity"]
    w = WheatCropModel(WheatParameters()).run(season, LAT).summary["days_to_maturity"]
    assert b < w


def test_phenology_ordered(model):
    res = model.run(_constant_season(17, 23, 9, 3, 6.0), LAT)
    assert list(res.daily["tt_cum"]) == sorted(res.daily["tt_cum"])
    assert (_stage_day(res, BarleyStage.HEADING)
            < _stage_day(res, BarleyStage.ANTHESIS)
            <= _stage_day(res, BarleyStage.MATURITY))


def test_heat_and_drought_reduce_yield(model):
    favourable = model.run(_constant_season(17, 23, 9, 3, 6.0), LAT).summary
    hot_dry = model.run(_constant_season(24, 33, 15, 10, 5.0), LAT).summary
    assert favourable["yield_kg_ha"] > hot_dry["yield_kg_ha"]
    assert hot_dry["grain_fill_heat_days"] > favourable["grain_fill_heat_days"]
