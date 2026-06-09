"""Behavioural tests for the yellow-pea process model."""

import numpy as np
import pandas as pd
import pytest

from canola_dt.simulation.pea_model import PeaCropModel, PeaParameters, PeaStage

LAT = 50.5


def _constant_season(tmean, tmax, tmin, precip_every, precip_mm, n=150):
    dates = pd.date_range("2022-05-01", periods=n, freq="D")
    precip = np.where(np.arange(n) % precip_every == 0, precip_mm, 0.0)
    return pd.DataFrame({
        "date": dates, "tmean_c": np.full(n, tmean, float), "tmax_c": np.full(n, tmax, float),
        "tmin_c": np.full(n, tmin, float), "precip_mm": precip,
    })


@pytest.fixture
def model():
    return PeaCropModel(PeaParameters())


def test_favourable_season_runs_to_maturity(model):
    s = model.run(_constant_season(16, 22, 8, 3, 6.0), LAT).summary
    assert s["reached_maturity"] is True
    assert 800 < s["yield_kg_ha"] < 5000
    assert 0.0 < s["harvest_index"] <= 0.48


def test_phenology_ordered(model):
    res = model.run(_constant_season(16, 22, 8, 3, 6.0), LAT)
    assert list(res.daily["tt_cum"]) == sorted(res.daily["tt_cum"])
    daily, start = res.daily, res.daily["date"].iloc[0]

    def day(stage):
        return int((daily[daily["stage"] >= int(stage)]["date"].iloc[0] - start).days)
    assert day(PeaStage.FLOWERING) < day(PeaStage.POD_FILL) <= day(PeaStage.MATURITY)


def test_heat_at_flowering_reduces_yield(model):
    # Peas abort flowers above 25 C -> a hot season yields less than a cool one.
    cool = model.run(_constant_season(16, 22, 8, 3, 6.0), LAT).summary
    hot = model.run(_constant_season(20, 30, 12, 3, 6.0), LAT).summary
    assert cool["yield_kg_ha"] > hot["yield_kg_ha"]
    assert hot["flowering_heat_days"] > cool["flowering_heat_days"]
