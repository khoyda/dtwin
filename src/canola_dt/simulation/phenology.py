"""Crop-timing (phenology) outputs derived from a simulated state trajectory.

Two deliverables:

1. :func:`stage_timeline` — *observed/simulated* timing: the calendar date and
   day-of-year on which each growth stage was first reached over the run.
2. :func:`forecast_stage_dates` — *forward* timing: given the crop's current
   cumulative GDD and date, predict when upcoming stages will be reached assuming
   an expected daily-GDD accrual rate. Useful for scheduling field operations
   (e.g. fungicide at early flowering, swathing/harvest readiness at maturity).
"""

from __future__ import annotations

import math

import pandas as pd

from canola_dt.constants import GrowthStage


def stage_timeline(trajectory: pd.DataFrame) -> pd.DataFrame:
    """First date each growth stage was reached in a simulated trajectory.

    Parameters
    ----------
    trajectory:
        Output of :meth:`canola_dt.simulation.growth.GrowthSimulator.run`, with
        ``date``, ``stage`` (:class:`GrowthStage`) and ``cum_gdd`` columns.

    Returns
    -------
    DataFrame with columns ``stage, stage_name, date, doy, cum_gdd, days_from_start``.
    """
    start = trajectory["date"].iloc[0]
    rows = []
    for stage, grp in trajectory.groupby("stage", sort=True):
        first = grp.iloc[0]
        rows.append(
            {
                "stage": int(stage),
                "stage_name": GrowthStage(int(stage)).name,
                "date": first["date"],
                "doy": int(first["date"].dayofyear),
                "cum_gdd": float(first["cum_gdd"]),
                "days_from_start": int((first["date"] - start).days),
            }
        )
    return pd.DataFrame(rows).sort_values("stage").reset_index(drop=True)


def expected_daily_gdd(trajectory: pd.DataFrame) -> float:
    """Mean per-day GDD accrual observed so far (simple forecast rate)."""
    if len(trajectory) < 2:
        return float("nan")
    span_days = (trajectory["date"].iloc[-1] - trajectory["date"].iloc[0]).days or 1
    return float(trajectory["cum_gdd"].iloc[-1]) / span_days


def forecast_stage_dates(
    current_cum_gdd: float,
    current_date: pd.Timestamp,
    stage_thresholds: dict[GrowthStage, float],
    daily_gdd_rate: float,
) -> pd.DataFrame:
    """Predict calendar dates for stages not yet reached.

    Parameters
    ----------
    current_cum_gdd:
        Cumulative GDD accrued to date.
    current_date:
        The "as-of" date the forecast is made from.
    stage_thresholds:
        Mapping of :class:`GrowthStage` -> cumulative-GDD entry threshold.
    daily_gdd_rate:
        Assumed future GDD accrual per day (e.g. from :func:`expected_daily_gdd`
        or a regional climatology). Must be > 0.

    Returns
    -------
    DataFrame with ``stage, stage_name, gdd_threshold, gdd_remaining,
    days_until, forecast_date`` for each upcoming stage, soonest first.
    """
    if daily_gdd_rate <= 0:
        raise ValueError("daily_gdd_rate must be positive to forecast timing")

    rows = []
    upcoming = {s: g for s, g in stage_thresholds.items() if g > current_cum_gdd}
    for stage, threshold in sorted(upcoming.items(), key=lambda kv: kv[1]):
        remaining = threshold - current_cum_gdd
        days = math.ceil(remaining / daily_gdd_rate)
        rows.append(
            {
                "stage": int(stage),
                "stage_name": GrowthStage(int(stage)).name,
                "gdd_threshold": float(threshold),
                "gdd_remaining": round(float(remaining), 1),
                "days_until": int(days),
                "forecast_date": current_date + pd.Timedelta(days=days),
            }
        )
    return pd.DataFrame(rows)
