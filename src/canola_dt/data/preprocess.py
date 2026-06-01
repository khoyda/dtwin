"""Clean and validate daily weather frames before feature engineering."""

from __future__ import annotations

import pandas as pd

from canola_dt.data.ingest import WEATHER_COLUMNS


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, de-duplicate, sort, and fill small gaps in a daily weather frame.

    - Drops duplicate dates (keeps first).
    - Reindexes to a continuous daily range and interpolates short gaps.
    - Derives ``tmean_c`` from tmin/tmax if missing.
    """
    df = df.copy()
    if "tmean_c" not in df or df["tmean_c"].isna().all():
        df["tmean_c"] = (df["tmin_c"] + df["tmax_c"]) / 2.0

    df = df[WEATHER_COLUMNS].drop_duplicates("date").sort_values("date")
    df = df.set_index("date").asfreq("D")

    # Temperatures interpolated linearly; precip gaps treated as dry days.
    df[["tmin_c", "tmax_c", "tmean_c"]] = df[["tmin_c", "tmax_c", "tmean_c"]].interpolate(
        limit=3
    )
    df["precip_mm"] = df["precip_mm"].fillna(0.0)

    return df.reset_index()
