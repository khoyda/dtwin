"""Statistics Canada crop-yield ingestion (Table 32-10-0359).

Downloads the full table via the StatCan Web Data Service, caches the zip, and
extracts canola **average yield (kg/ha)** by province and year — the training
target for the yield model.

WDS full-table endpoint returns a JSON pointer to the CSV zip::

    https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/<PID>/en
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

WDS_FULL_CSV = "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{pid}/en"
_USER_AGENT = {"User-Agent": "canola-dt/0.1 (research)"}


def _http_get(url: str) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_USER_AGENT), timeout=120
    ).read()


def download_table(pid: str, cache_dir: str | Path) -> Path:
    """Download and cache the table zip; returns the local zip path."""
    zpath = Path(cache_dir) / f"{pid}-eng.zip"
    if zpath.exists():
        return zpath
    meta = json.loads(_http_get(WDS_FULL_CSV.format(pid=pid)))
    if meta.get("status") != "SUCCESS":
        raise RuntimeError(f"StatCan WDS error for pid {pid}: {meta}")
    zpath.parent.mkdir(parents=True, exist_ok=True)
    zpath.write_bytes(_http_get(meta["object"]))
    return zpath


def load_canola_yield(
    pid: str,
    cache_dir: str | Path,
    provinces: list[str],
    crop: str = "Canola (rapeseed)",
    yield_disposition: str = "Average yield (kilograms per hectare)",
) -> pd.DataFrame:
    """Return tidy canola yield: columns ``province, year, yield_kg_ha``.

    Filtered to ``provinces``, deduplicated on (province, year), missing values
    (``VALUE`` NaN, e.g. suppressed estimates) dropped.
    """
    zpath = download_table(pid, cache_dir)
    with zipfile.ZipFile(zpath) as z:
        data_csv = next(
            n for n in z.namelist() if n.lower().endswith(".csv") and "metadata" not in n.lower()
        )
        df = pd.read_csv(z.open(data_csv), low_memory=False)

    mask = (
        (df["Type of crop"] == crop)
        & (df["Harvest disposition"] == yield_disposition)
        & (df["GEO"].isin(provinces))
        & (df["VALUE"].notna())
    )
    out = (
        df.loc[mask, ["GEO", "REF_DATE", "VALUE"]]
        .rename(columns={"GEO": "province", "REF_DATE": "year", "VALUE": "yield_kg_ha"})
        .astype({"year": int, "yield_kg_ha": float})
        .drop_duplicates(["province", "year"])
        .sort_values(["province", "year"])
        .reset_index(drop=True)
    )
    return out
