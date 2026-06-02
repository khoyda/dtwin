"""Sub-provincial validation against Saskatchewan RM-level canola yields (SCIC).

The provincial calibration hit a skill ceiling (anomaly correlation ~0.39) because a
provincial yield averages thousands of fields, smoothing out the weather signal a point
station sees. This module tests the hypothesis directly: match each ECCC station to its
**Rural Municipality (RM)** and compare the station's *simulated* canola yield to that
**local** RM's *observed* yield — a scale at which local weather should map to local yield.

Data sources (Saskatchewan):
* RM canola yields — Saskatchewan Dashboard "RM Yields" export (SCIC + Crop Report),
  reported in bushels/acre.
* RM centroids — Government of Saskatchewan ArcGIS "rural municipality" feature service.
"""

from __future__ import annotations

import csv
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from canola_dt import calibration as cal
from canola_dt.config import Config
from canola_dt.data import eccc
from canola_dt.data.aafc import CANOLA_BU_AC_TO_KG_HA
from canola_dt.simulation.process_model import CanolaCropModel, CanolaParameters

RM_YIELDS_CSV = "https://dashboard.saskatchewan.ca/export/rm-yields-data/4950.csv"
RM_CENTROIDS_QUERY = (
    "https://services9.arcgis.com/WJsMXAAF3vSdDYis/arcgis/rest/services/"
    "SaskAdmin_2016_rural_municipality/FeatureServer/0/query"
    "?where=1%3D1&outFields=RMNO,RMNM&returnCentroid=true&returnGeometry=false"
    "&outSR=4326&f=json"
)
_UA = {"User-Agent": "Mozilla/5.0 (canola-dt research)"}


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=90).read()


# --- data loaders (cached) ---------------------------------------------------

def load_rm_canola_yields(cache_dir: str | Path) -> pd.DataFrame:
    """SK RM canola yields -> columns ``rmno, year, yield_kg_ha`` (from bu/ac)."""
    path = Path(cache_dir) / "sk_rm_yields.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(RM_YIELDS_CSV))
    df = pd.read_csv(path)
    out = df[["Year", "RM", "Canola"]].dropna(subset=["Canola"]).copy()
    out.columns = ["year", "rmno", "yield_bu_ac"]
    out["year"] = out["year"].astype(int)
    out["rmno"] = out["rmno"].astype(int)
    out["yield_kg_ha"] = out["yield_bu_ac"] * CANOLA_BU_AC_TO_KG_HA
    return out[["rmno", "year", "yield_kg_ha"]].reset_index(drop=True)


def load_rm_centroids(cache_dir: str | Path) -> pd.DataFrame:
    """SK RM centroids -> columns ``rmno, lat, lon`` (cached from ArcGIS)."""
    path = Path(cache_dir) / "sk_rm_centroids.csv"
    if not path.exists():
        data = json.loads(_get(RM_CENTROIDS_QUERY))
        rows = []
        for f in data.get("features", []):
            c = f.get("centroid")
            if c:
                rows.append((int(f["attributes"]["RMNO"]), round(c["y"], 4), round(c["x"], 4)))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rmno", "lat", "lon"])
            w.writerows(rows)
    return pd.read_csv(path)


# --- spatial matching --------------------------------------------------------

def _planar_dist(lat1, lon1, lat2, lon2) -> float:
    scale = math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(lat1 - lat2, (lon1 - lon2) * scale)


def nearest_rm(lat: float, lon: float, centroids: pd.DataFrame) -> int:
    d = centroids.apply(lambda r: _planar_dist(lat, lon, r["lat"], r["lon"]), axis=1)
    return int(centroids.loc[d.idxmin(), "rmno"])


# --- validation --------------------------------------------------------------

def build_station_rm_pairs(cfg: Config, params: CanolaParameters | None = None) -> pd.DataFrame:
    """Pair each SK station-year's simulated yield with its nearest RM's observed yield.

    Returns ``station_id, rmno, year, sim_yield, rm_yield`` (kg/ha).
    """
    params = params or CanolaParameters.from_calibrated(cfg)
    frames = cal.load_season_frames(cfg)
    centroids = load_rm_centroids(cfg.path("data_external"))
    rm_yields = load_rm_canola_yields(cfg.path("data_external"))
    stations = eccc.station_map(cfg)

    # Nearest RM for each SK station.
    station_rm = {
        sid: nearest_rm(info["lat"], info["lon"], centroids)
        for sid, info in stations.items()
        if info["province"] == "Saskatchewan"
    }
    rm_lookup = rm_yields.set_index(["rmno", "year"])["yield_kg_ha"].to_dict()

    rows = []
    for (province, station_id, year), (frame, lat) in frames.items():
        if province != "Saskatchewan":
            continue
        rmno = station_rm.get(station_id)
        obs = rm_lookup.get((rmno, year))
        if obs is None:
            continue
        sim = CanolaCropModel(params).run(frame, lat).summary["yield_kg_ha"]
        rows.append({"station_id": station_id, "rmno": rmno, "year": year,
                     "sim_yield": sim, "rm_yield": obs})
    return pd.DataFrame(rows)


def _detrended_anomaly_corr(pairs: pd.DataFrame, obs_col: str, group: str) -> float:
    """Pooled correlation of (sim - group mean) vs (obs - per-group linear trend)."""
    sim_anom, obs_anom = [], []
    for _, g in pairs.groupby(group):
        if len(g) < 3:
            continue
        slope, intercept = np.polyfit(g["year"], g[obs_col], 1)
        obs_anom.extend(g[obs_col] - (intercept + slope * g["year"]))
        sim_anom.extend(g["sim_yield"] - g["sim_yield"].mean())
    if len(sim_anom) < 3:
        return float("nan")
    return float(np.corrcoef(sim_anom, obs_anom)[0, 1])


def local_vs_provincial(cfg: Config, params: CanolaParameters | None = None) -> dict:
    """Compare station-sim skill against LOCAL RM yields vs the SK PROVINCIAL yield.

    Both use the same SK stations and the same detrending, so the only difference is
    the spatial scale of the yield target.
    """
    pairs = build_station_rm_pairs(cfg, params)

    # Provincial SK target joined to the same station-years.
    prov = cal.load_targets(cfg)
    prov_sk = prov[prov["province"] == "Saskatchewan"].set_index("year")["yield_kg_ha"].to_dict()
    pairs = pairs.assign(prov_yield=pairs["year"].map(prov_sk))
    paired_prov = pairs.dropna(subset=["prov_yield"])

    return {
        "n_pairs": int(len(pairs)),
        "n_stations": int(pairs["station_id"].nunique()),
        "local_anomaly_corr": _detrended_anomaly_corr(pairs, "rm_yield", "station_id"),
        "provincial_anomaly_corr": _detrended_anomaly_corr(paired_prov, "prov_yield", "station_id"),
        "station_rm": pairs.groupby("station_id")["rmno"].first().to_dict(),
    }
