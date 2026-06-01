"""Load raw data sources into tidy pandas frames.

Real connectors (ECCC, NASA POWER, StatCan, SoilGrids) are stubbed here with the
expected output schema documented in each docstring. ``synthetic_weather`` provides
a runnable stand-in so the pipeline works end-to-end before real data is wired up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical daily weather schema consumed downstream.
WEATHER_COLUMNS = ["date", "tmin_c", "tmax_c", "tmean_c", "precip_mm"]


def load_weather_csv(path: str | Path) -> pd.DataFrame:
    """Load a daily weather CSV with columns :data:`WEATHER_COLUMNS`.

    Expected source: ECCC station export or NASA POWER point download, pre-renamed.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(WEATHER_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"weather file {path} missing columns: {sorted(missing)}")
    return df[WEATHER_COLUMNS].sort_values("date").reset_index(drop=True)


def load_yield_history(path: str | Path) -> pd.DataFrame:
    """Load historical field-season yields.

    Expected columns: ``field_id, year, yield_kg_ha`` (+ optional management cols).
    Source: StatCan Table 32-10-0359 (regional) or farm records (field-level).
    """
    return pd.read_csv(path)


def synthetic_weather(
    year: int = 2023,
    seeding_doy: int = 130,
    n_days: int = 130,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a plausible Prairie growing-season daily weather frame.

    Useful for demos and tests. Temperatures follow a seasonal arch; precipitation
    is sparse and skewed, roughly mimicking a semi-arid Prairie summer.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=seeding_doy - 1)
    dates = pd.date_range(start, periods=n_days, freq="D")

    t = np.arange(n_days)
    # Seasonal mean temperature: rises to a midsummer peak then declines.
    seasonal = 18.0 + 8.0 * np.sin(np.pi * t / n_days)
    tmean = seasonal + rng.normal(0, 2.5, n_days)
    tmax = tmean + rng.uniform(4, 9, n_days)
    tmin = tmean - rng.uniform(4, 8, n_days)

    # ~30% of days have rain; amounts are exponentially distributed.
    wet = rng.random(n_days) < 0.30
    precip = np.where(wet, rng.exponential(6.0, n_days), 0.0)

    return pd.DataFrame(
        {
            "date": dates,
            "tmin_c": tmin.round(2),
            "tmax_c": tmax.round(2),
            "tmean_c": tmean.round(2),
            "precip_mm": precip.round(2),
        }
    )
