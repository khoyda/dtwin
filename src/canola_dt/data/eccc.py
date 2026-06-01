"""Environment & Climate Change Canada (ECCC) daily weather ingestion.

Downloads daily climate data via the public bulk-data endpoint and normalizes it
to the canonical schema used downstream (:data:`canola_dt.data.ingest.WEATHER_COLUMNS`).
Station-year CSVs are cached on disk so repeated runs hit the network only once.

Bulk-data endpoint (one calendar year per request, ``timeframe=2`` = daily)::

    https://climate.weather.gc.ca/climate_data/bulk_data_e.html
        ?format=csv&stationID=<ID>&Year=<Y>&Month=1&Day=1&timeframe=2&submit=Download
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

from canola_dt.data.ingest import WEATHER_COLUMNS

BULK_URL = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    "?format=csv&stationID={station_id}&Year={year}&Month=1&Day=1"
    "&timeframe=2&submit=Download"
)
INVENTORY_URL = (
    "https://collaboration.cmc.ec.gc.ca/cmc/climate/Get_More_Data_Plus_de_donnees/"
    "Station%20Inventory%20EN.csv"
)
_USER_AGENT = {"User-Agent": "canola-dt/0.1 (research)"}


def _http_get(url: str) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_USER_AGENT), timeout=60
    ).read()


def _find_col(columns, *needles: str) -> str:
    """Return the first column whose name contains all ``needles`` (case-insensitive).

    ECCC daily CSVs encode the degree symbol inconsistently (e.g. ``Max Temp (\xb0C)``),
    so we match on stable substrings like "Max Temp" rather than exact headers.
    """
    low = {c: c.lower() for c in columns}
    for c, cl in low.items():
        if all(n.lower() in cl for n in needles):
            return c
    raise KeyError(f"no column matching {needles} in {list(columns)}")


def load_station_inventory(cache_dir: str | Path) -> pd.DataFrame:
    """Download (and cache) the ECCC station inventory; 3 preamble lines skipped."""
    path = Path(cache_dir) / "station_inventory.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_http_get(INVENTORY_URL))
    return pd.read_csv(path, skiprows=3, low_memory=False)


def fetch_daily(station_id: int, year: int, cache_dir: str | Path) -> pd.DataFrame:
    """Fetch one station-year of daily weather, normalized and cached.

    Returns a frame with :data:`WEATHER_COLUMNS`
    (``date, tmin_c, tmax_c, tmean_c, precip_mm``). Missing values are preserved
    (NaN) for the preprocessing/cleaning step to handle.
    """
    cache = Path(cache_dir) / f"{station_id}_{year}.csv"
    if cache.exists():
        raw = cache.read_bytes()
    else:
        raw = _http_get(BULK_URL.format(station_id=station_id, year=year))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)

    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", low_memory=False)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[_find_col(df.columns, "Date/Time")]),
            "tmax_c": pd.to_numeric(df[_find_col(df.columns, "Max Temp")], errors="coerce"),
            "tmin_c": pd.to_numeric(df[_find_col(df.columns, "Min Temp")], errors="coerce"),
            "tmean_c": pd.to_numeric(df[_find_col(df.columns, "Mean Temp")], errors="coerce"),
            "precip_mm": pd.to_numeric(
                df[_find_col(df.columns, "Total Precip")], errors="coerce"
            ),
        }
    )
    return out[WEATHER_COLUMNS].sort_values("date").reset_index(drop=True)


def growing_season_weather(
    station_id: int,
    year: int,
    cache_dir: str | Path,
    season_start: tuple[int, int] = (5, 1),
    season_end: tuple[int, int] = (9, 30),
) -> pd.DataFrame:
    """Daily weather restricted to the growing-season window for one station-year."""
    df = fetch_daily(station_id, year, cache_dir)
    start = pd.Timestamp(year=year, month=season_start[0], day=season_start[1])
    end = pd.Timestamp(year=year, month=season_end[0], day=season_end[1])
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


def season_completeness(weather: pd.DataFrame) -> float:
    """Fraction of growing-season days with both mean temp and precip present."""
    if weather.empty:
        return 0.0
    ok = weather["tmean_c"].notna() & weather["precip_mm"].notna()
    return float(ok.mean())
