"""Season-level feature engineering from daily weather.

These features feed the ML yield model and double as drivers for the process
simulation. They encode the agronomic levers that matter most for Prairie canola:
heat accumulation (GDD), reproductive heat stress, and water supply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canola_dt import constants


def growing_degree_days(
    tmean_c: pd.Series,
    base_temp_c: float = constants.GDD_BASE_TEMP_C,
    cap_temp_c: float = constants.GDD_CAP_TEMP_C,
) -> pd.Series:
    """Daily GDD = max(0, min(tmean, cap) - base). Returns a per-day series."""
    capped = tmean_c.clip(upper=cap_temp_c)
    return (capped - base_temp_c).clip(lower=0.0)


def cumulative_gdd(tmean_c: pd.Series, **kwargs) -> pd.Series:
    """Cumulative GDD over the season."""
    return growing_degree_days(tmean_c, **kwargs).cumsum()


def heat_stress_days(
    tmax_c: pd.Series,
    threshold_c: float = constants.HEAT_STRESS_THRESHOLD_C,
) -> pd.Series:
    """Boolean series marking days exceeding the flowering heat-stress threshold."""
    return tmax_c > threshold_c


def season_features(
    weather: pd.DataFrame,
    base_temp_c: float = constants.GDD_BASE_TEMP_C,
    cap_temp_c: float = constants.GDD_CAP_TEMP_C,
    heat_threshold_c: float = constants.HEAT_STRESS_THRESHOLD_C,
) -> dict[str, float]:
    """Aggregate a daily weather frame into one row of season-level features.

    Returns a flat dict suitable for assembling a training/inference table.
    """
    gdd = growing_degree_days(weather["tmean_c"], base_temp_c, cap_temp_c)
    stress = heat_stress_days(weather["tmax_c"], heat_threshold_c)

    return {
        "total_gdd": float(gdd.sum()),
        "total_precip_mm": float(weather["precip_mm"].sum()),
        "mean_tmax_c": float(weather["tmax_c"].mean()),
        "mean_tmin_c": float(weather["tmin_c"].mean()),
        "heat_stress_days": int(stress.sum()),
        "dry_days": int((weather["precip_mm"] <= 0.1).sum()),
        "max_dry_spell": int(_max_dry_spell(weather["precip_mm"])),
        "season_length_days": int(len(weather)),
    }


def _max_dry_spell(precip_mm: pd.Series, wet_threshold_mm: float = 0.1) -> int:
    """Longest run of consecutive days with precipitation below ``wet_threshold_mm``."""
    dry = (precip_mm <= wet_threshold_mm).to_numpy()
    best = run = 0
    for d in dry:
        run = run + 1 if d else 0
        best = max(best, run)
    return best
