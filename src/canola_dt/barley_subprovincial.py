"""Sub-provincial validation of the spring-barley model vs SK RM-level yields.

Mirrors :mod:`canola_dt.subprovincial` / :mod:`canola_dt.wheat_subprovincial` for barley:
matches each ECCC station to its nearest Rural Municipality and compares the station's
*simulated* barley yield to that RM's *observed* SCIC yield (the dashboard "Barley" column),
testing whether local matching beats the provincial-scale skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canola_dt import barley_calibration as bc
from canola_dt import calibration as cal
from canola_dt import subprovincial as sp
from canola_dt.config import Config
from canola_dt.data import eccc
from canola_dt.data.aafc import BARLEY_BU_AC_TO_KG_HA
from canola_dt.simulation.barley_model import BarleyCropModel, BarleyParameters

RM_BARLEY_COLUMN = "Barley"


def load_rm_barley_yields(cache_dir) -> pd.DataFrame:
    """SK RM barley yields -> ``rmno, year, yield_kg_ha`` (48-lb bushel)."""
    return sp.load_rm_crop_yields(cache_dir, RM_BARLEY_COLUMN, BARLEY_BU_AC_TO_KG_HA)


def build_station_rm_pairs(cfg: Config, params: BarleyParameters | None = None) -> pd.DataFrame:
    """Pair each SK station-year's simulated barley yield with its nearest RM's yield."""
    params = params or BarleyParameters.from_calibrated(cfg)
    frames = cal.load_season_frames(cfg)
    centroids = sp.load_rm_centroids(cfg.path("data_external"))
    rm_yields = load_rm_barley_yields(cfg.path("data_external"))
    stations = eccc.station_map(cfg)

    station_rm = {
        sid: sp.nearest_rm(info["lat"], info["lon"], centroids)
        for sid, info in stations.items() if info["province"] == "Saskatchewan"
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
        sim = BarleyCropModel(params).run(frame, lat).summary["yield_kg_ha"]
        rows.append({"station_id": station_id, "rmno": rmno, "year": year,
                     "sim_yield": sim, "rm_yield": obs})
    return pd.DataFrame(rows)


def local_vs_provincial(cfg: Config, params: BarleyParameters | None = None) -> dict:
    """Compare station-sim skill vs LOCAL RM barley yields and vs the PROVINCIAL yield."""
    pairs = build_station_rm_pairs(cfg, params)

    prov = bc.load_targets(cfg)
    prov_sk = prov[prov["province"] == "Saskatchewan"].set_index("year")["yield_kg_ha"].to_dict()
    pairs = pairs.assign(prov_yield=pairs["year"].map(prov_sk))
    paired_prov = pairs.dropna(subset=["prov_yield"])

    return {
        "n_pairs": int(len(pairs)),
        "n_stations": int(pairs["station_id"].nunique()),
        "local_anomaly_corr": sp._detrended_anomaly_corr(pairs, "rm_yield", "station_id"),
        "provincial_anomaly_corr": sp._detrended_anomaly_corr(paired_prov, "prov_yield", "station_id"),
        "station_rm": pairs.groupby("station_id")["rmno"].first().to_dict(),
        "pairs": pairs,
    }
