"""Calibrate the spring-barley process model against trend-adjusted StatCan yields.

Reuses the crop-agnostic machinery from :mod:`canola_dt.calibration` (weather frames,
detrending, anomaly metrics, per-province offsets) with the barley crop model and the
StatCan "Barley" yield series. Same three-step method: pattern (kl, heat) -> level (rue)
-> per-province offsets.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from canola_dt import calibration as cal
from canola_dt.config import Config, load_config
from canola_dt.data import statcan
from canola_dt.simulation.barley_model import CALIBRATABLE, BarleyCropModel, BarleyParameters


def load_targets(cfg: Config) -> pd.DataFrame:
    ds = cfg["data_sources"]
    obs = statcan.load_canola_yield(
        pid=ds["statcan"]["table_pid"],
        cache_dir=cfg.path("data_raw") / "statcan",
        provinces=ds["statcan"]["provinces"],
        crop=ds["statcan"]["barley_crop"],
        yield_disposition=ds["statcan"]["yield_disposition"],
    )
    obs = obs[(obs["year"] >= ds["start_year"]) & (obs["year"] <= ds["end_year"])]
    obs = obs[~obs["year"].isin(ds.get("exclude_years", []) or [])]
    return cal.technology_detrend(obs, ref_year=ds["end_year"])


def simulate_all(frames: dict, params: BarleyParameters) -> pd.DataFrame:
    rows = []
    for (province, station_id, year), (frame, lat) in frames.items():
        res = BarleyCropModel(params).run(frame, lat)
        rows.append({"province": province, "station_id": station_id, "year": year,
                     "sim_yield": res.summary["yield_kg_ha"]})
    per_station = pd.DataFrame(rows)
    return (per_station.groupby(["province", "year"], as_index=False)
            .agg(sim_yield=("sim_yield", "mean"), n_stations=("station_id", "nunique")))


def evaluate(frames: dict, targets: pd.DataFrame, params: BarleyParameters):
    sim = simulate_all(frames, params)
    merged = sim.merge(targets[["province", "year", "adjusted", "yield_kg_ha", "trend"]],
                       on=["province", "year"])
    return cal.calibration_metrics(merged), merged


def calibrate(cfg: Config, base: BarleyParameters, grid: dict | None = None) -> dict:
    grid = grid or cal.DEFAULT_GRID
    frames = cal.load_season_frames(cfg)
    targets = load_targets(cfg)

    rows, best = [], None
    for kl, hs in itertools.product(grid["kl"], grid["hi_heat_sensitivity"]):
        ov = {"kl": kl, "hi_heat_sensitivity": hs}
        m, _ = evaluate(frames, targets, base.with_overrides(ov))
        rows.append({**ov, "anomaly_corr": m["anomaly_corr"], "anomaly_rmse": m["anomaly_rmse"]})
        corr = -1.0 if m["anomaly_corr"] != m["anomaly_corr"] else m["anomaly_corr"]
        if best is None or corr > best[1]:
            best = (ov, corr)
    pattern = best[0]

    _, merged0 = evaluate(frames, targets, base.with_overrides(pattern))
    sim_std, obs_std = cal._anom_std(merged0)
    rue_star = float(np.clip(
        base.rue * merged0["adjusted"].mean() / max(merged0["sim_yield"].mean(), 1e-6), 1.0, 3.5))
    params = {"rue": round(rue_star, 2), **pattern}

    metrics, merged = evaluate(frames, targets, base.with_overrides(params))
    offsets = cal.province_offsets(merged)
    pattern_table = (pd.DataFrame(rows).sort_values("anomaly_corr", ascending=False)
                     .reset_index(drop=True))
    return {
        "params": params, "metrics": metrics, "offsets": offsets,
        "corrected": cal.corrected_metrics(merged, offsets),
        "diagnostics": {"sim_anom_std": round(sim_std, 1), "obs_anom_std": round(obs_std, 1),
                        "volatility_ratio": round(sim_std / max(obs_std, 1e-6), 2)},
        "pattern_table": pattern_table, "n": len(frames),
    }


def save_calibrated(cfg: Config, overrides: dict):
    path = cfg.path("artifacts") / "barley_calibrated_params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: float(v) for k, v in overrides.items() if k in CALIBRATABLE},
                               indent=2))
    return path


if __name__ == "__main__":
    cfg = load_config()
    out = calibrate(cfg, BarleyParameters.from_config(cfg))
    print("calibrated:", out["params"], "| metrics:", out["metrics"])
