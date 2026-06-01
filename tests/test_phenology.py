"""Tests for crop-timing (phenology) outputs."""

import pandas as pd

from canola_dt.constants import GrowthStage
from canola_dt.simulation.phenology import (
    expected_daily_gdd,
    forecast_stage_dates,
    stage_timeline,
)


def _toy_trajectory() -> pd.DataFrame:
    dates = pd.date_range("2023-05-10", periods=4, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "cum_gdd": [50.0, 130.0, 200.0, 360.0],
            "stage": [
                GrowthStage.EMERGENCE,
                GrowthStage.EMERGENCE,
                GrowthStage.ROSETTE,
                GrowthStage.ROSETTE,
            ],
        }
    )


def test_stage_timeline_first_occurrence():
    tl = stage_timeline(_toy_trajectory())
    rosette = tl[tl["stage_name"] == "ROSETTE"].iloc[0]
    assert rosette["date"] == pd.Timestamp("2023-05-12")
    assert rosette["days_from_start"] == 2
    assert rosette["cum_gdd"] == 200.0


def test_expected_daily_gdd():
    # 360 GDD accrued over 3 days span -> 120 GDD/day.
    assert expected_daily_gdd(_toy_trajectory()) == 120.0


def test_forecast_only_upcoming_stages_sorted():
    thresholds = {
        GrowthStage.ROSETTE: 350.0,
        GrowthStage.BOLTING: 600.0,
        GrowthStage.FLOWERING: 900.0,
    }
    fc = forecast_stage_dates(
        current_cum_gdd=360.0,
        current_date=pd.Timestamp("2023-05-13"),
        stage_thresholds=thresholds,
        daily_gdd_rate=120.0,
    )
    # ROSETTE already passed; only BOLTING and FLOWERING remain, soonest first.
    assert list(fc["stage_name"]) == ["BOLTING", "FLOWERING"]
    bolting = fc.iloc[0]
    assert bolting["gdd_remaining"] == 240.0
    assert bolting["days_until"] == 2  # ceil(240/120)
    assert bolting["forecast_date"] == pd.Timestamp("2023-05-15")
