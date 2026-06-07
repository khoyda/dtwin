"""NASA POWER gridded daily weather ingestion (point query on a ~0.5° grid).

Lets the crop models run at *any* lat/lon (e.g. a Rural-Municipality centroid) without
needing a nearby ECCC station — the basis for sub-provincial work in provinces that lack
dense station coverage. Normalizes to the canonical daily schema
(:data:`canola_dt.data.ingest.WEATHER_COLUMNS`) and caches per grid-cell-year.

API (no key required)::

    https://power.larc.nasa.gov/api/temporal/daily/point
        ?parameters=T2M_MAX,T2M_MIN,T2M,PRECTOTCORR&community=AG
        &latitude=<lat>&longitude=<lon>&start=YYYYMMDD&end=YYYYMMDD&format=JSON
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from canola_dt.data.ingest import WEATHER_COLUMNS

POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=T2M_MAX,T2M_MIN,T2M,PRECTOTCORR&community=AG"
    "&latitude={lat}&longitude={lon}&start={start}&end={end}&format=JSON"
)
_USER_AGENT = {"User-Agent": "canola-dt/0.1 (research)"}
_FILL = -999.0  # NASA POWER missing-value sentinel


def _http_get(url: str) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_USER_AGENT), timeout=90
    ).read()


def parse_power_json(data: dict) -> pd.DataFrame:
    """Convert a NASA POWER daily-point JSON payload to the canonical weather frame."""
    p = data["properties"]["parameter"]
    dates = sorted(p["T2M"].keys())
    rows = {
        "date": pd.to_datetime(dates, format="%Y%m%d"),
        "tmax_c": [p["T2M_MAX"][d] for d in dates],
        "tmin_c": [p["T2M_MIN"][d] for d in dates],
        "tmean_c": [p["T2M"][d] for d in dates],
        "precip_mm": [p["PRECTOTCORR"][d] for d in dates],
    }
    df = pd.DataFrame(rows)
    for c in ("tmax_c", "tmin_c", "tmean_c"):
        df.loc[df[c] <= _FILL, c] = pd.NA
    df["precip_mm"] = df["precip_mm"].where(df["precip_mm"] > _FILL, 0.0)
    return df[WEATHER_COLUMNS]


def fetch_daily(lat: float, lon: float, year: int, cache_dir: str | Path,
                season_start: tuple[int, int] = (5, 1),
                season_end: tuple[int, int] = (10, 31)) -> pd.DataFrame:
    """Fetch one grid-cell-year of daily weather (growing-season window), cached.

    Cache key rounds lat/lon to 0.25° so nearby points share the (coarse-grid) cell.
    """
    glat, glon = round(lat * 4) / 4, round(lon * 4) / 4
    cache = Path(cache_dir) / f"{glat}_{glon}_{year}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
    else:
        start = f"{year}{season_start[0]:02d}{season_start[1]:02d}"
        end = f"{year}{season_end[0]:02d}{season_end[1]:02d}"
        data = json.loads(_http_get(POWER_URL.format(lat=glat, lon=glon, start=start, end=end)))
        df = parse_power_json(data)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def growing_season_weather(lat: float, lon: float, year: int, cache_dir: str | Path,
                           **kwargs) -> pd.DataFrame:
    """Continuous daily growing-season frame for the crop models (gaps interpolated)."""
    df = fetch_daily(lat, lon, year, cache_dir, **kwargs).set_index("date").asfreq("D")
    df[["tmin_c", "tmax_c", "tmean_c"]] = df[["tmin_c", "tmax_c", "tmean_c"]].interpolate(limit=5)
    df["precip_mm"] = df["precip_mm"].fillna(0.0)
    return df.reset_index()
