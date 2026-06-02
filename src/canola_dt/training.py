"""Assemble a real training set (ECCC weather + StatCan/AAFC yields) and fit the model.

For each (province, year) in the configured window we:
  1. pull growing-season daily weather from the province's ECCC station,
  2. run the digital twin (no model) to derive the *same* season + simulation
     features the twin produces at inference time, and
  3. join the StatCan canola yield (kg/ha) as the training target.

Optionally, an AAFC/SCIC sub-provincial yield CSV is folded in (province-aggregated)
when ``data_sources.aafc.yield_csv`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from canola_dt import calibration as cal
from canola_dt.config import Config, load_config
from canola_dt.data import aafc, eccc, statcan
from canola_dt.models.yield_model import YieldModel
from canola_dt.simulation.process_model import CanolaCropModel, CanolaParameters
from canola_dt.simulation.twin import CanolaDigitalTwin

MIN_SEASON_COMPLETENESS = 0.80  # drop province-years with too much missing weather

# APSIM-style process-model summary outputs surfaced as ML features (prefix pm_).
PM_SUMMARY_KEYS = {
    "yield_kg_ha": "pm_yield",
    "total_biomass_g_m2": "pm_biomass",
    "harvest_index": "pm_hi",
    "max_lai": "pm_max_lai",
    "days_to_flowering": "pm_days_to_flowering",
    "days_to_maturity": "pm_days_to_maturity",
    "flowering_heat_days": "pm_flower_heat",
    "mean_flowering_water_stress": "pm_water_stress",
    "season_transp_mm": "pm_transp",
    "season_soil_evap_mm": "pm_soil_evap",
}


@dataclass
class TrainingData:
    X: pd.DataFrame          # feature matrix
    y: pd.Series             # yield_kg_ha
    meta: pd.DataFrame       # province, year, yield_source, completeness


def build_weather_features(cfg: Config) -> pd.DataFrame:
    """Twin-derived features averaged across each province's stations, per year.

    Multiple stations per province are reduced to a province-year mean, mirroring
    the spatial average that a provincial yield statistic represents.
    """
    ds = cfg["data_sources"]
    cache = cfg.path("data_raw") / "eccc"
    s_start = (ds["season_start"]["month"], ds["season_start"]["day"])
    s_end = (ds["season_end"]["month"], ds["season_end"]["day"])
    twin = CanolaDigitalTwin(cfg)  # no model: used purely for feature assembly

    rows: list[dict] = []
    for station_id, info in eccc.station_map(cfg).items():
        for year in range(ds["start_year"], ds["end_year"] + 1):
            weather = eccc.growing_season_weather(
                int(station_id), year, cache, season_start=s_start, season_end=s_end
            )
            completeness = eccc.season_completeness(weather)
            if completeness < MIN_SEASON_COMPLETENESS or weather.empty:
                continue
            feats = twin.run(weather).features
            feats |= {
                "province": info["province"],
                "year": year,
                "station_id": int(station_id),
                "completeness": round(completeness, 3),
            }
            rows.append(feats)

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c not in {"province", "year", "station_id"}]
    agg = df.groupby(["province", "year"], as_index=False)[feat_cols].mean()
    agg["n_stations"] = (
        df.groupby(["province", "year"])["station_id"].nunique().reset_index(drop=True)
    )
    return agg

def build_process_features(cfg: Config, params: CanolaParameters | None = None) -> pd.DataFrame:
    """Calibrated APSIM-style process-model outputs, averaged per province-year.

    Runs the mechanistic crop model per station-year (May–Oct window) and aggregates
    its summary (simulated yield, biomass, LAI, harvest index, phenology timing, water
    stress, water fluxes) to a province-year mean — the same spatial aggregation used
    for the weather features. These ``pm_*`` columns couple the process model into the
    statistical model.
    """
    params = params or CanolaParameters.from_calibrated(cfg)
    frames = cal.load_season_frames(cfg)

    rows: list[dict] = []
    for (province, _station_id, year), (frame, lat) in frames.items():
        s = CanolaCropModel(params).run(frame, lat).summary
        row = {"province": province, "year": year,
               "pm_frac_matured": float(bool(s["reached_maturity"]))}
        for key, name in PM_SUMMARY_KEYS.items():
            v = s.get(key)
            row[name] = float(v) if v is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    agg = df.groupby(["province", "year"], as_index=False).mean(numeric_only=True)
    pm_cols = [c for c in agg.columns if c.startswith("pm_")]
    # e.g. days_to_maturity is NaN when a station never matured -> fill with column mean.
    agg[pm_cols] = agg[pm_cols].fillna(agg[pm_cols].mean())
    return agg


def build_training_data(cfg: Config | None = None, include_process: bool = True) -> TrainingData:
    """Join weather features (+ optional process-model features) to StatCan yields."""
    cfg = cfg or load_config()
    ds = cfg["data_sources"]
    feats = build_weather_features(cfg)
    if include_process:
        feats = feats.merge(build_process_features(cfg), on=["province", "year"], how="left")

    yields = statcan.load_canola_yield(
        pid=ds["statcan"]["table_pid"],
        cache_dir=cfg.path("data_raw") / "statcan",
        provinces=ds["statcan"]["provinces"],
        crop=ds["statcan"]["crop"],
        yield_disposition=ds["statcan"]["yield_disposition"],
    )
    yields = yields.assign(yield_source="StatCan")

    # Optional AAFC/SCIC sub-provincial yields, aggregated to province-year means.
    aafc_csv = ds.get("aafc", {}).get("yield_csv")
    if aafc_csv:
        a = aafc.load_region_yield(aafc_csv)
        a = (
            a.dropna(subset=["province"])
            .groupby(["province", "year"], as_index=False)["yield_kg_ha"]
            .mean()
            .assign(yield_source="AAFC")
        )
        # Prefer StatCan where both exist; append AAFC-only province-years.
        key = ["province", "year"]
        merged = a.merge(yields[key], on=key, how="left", indicator=True)
        a_only = a[merged["_merge"] == "left_only"]
        yields = pd.concat([yields, a_only], ignore_index=True)

    df = feats.merge(yields, on=["province", "year"], how="inner")

    meta_cols = ["province", "year", "n_stations", "completeness", "yield_source"]
    # Weather features + `year` as a trend proxy: canola yields carry a strong
    # upward technology/genetics signal over time that weather alone can't explain.
    weather_feats = [
        c for c in feats.columns
        if c not in {"province", "year", "n_stations", "completeness"}
    ]
    feature_cols = weather_feats + ["year"]
    return TrainingData(
        X=df[feature_cols].reset_index(drop=True),
        y=df["yield_kg_ha"].reset_index(drop=True),
        meta=df[meta_cols].reset_index(drop=True),
    )


def train(cfg: Config | None = None) -> tuple[YieldModel, dict, TrainingData]:
    """Build the dataset and fit the configured model; returns (model, metrics, data)."""
    cfg = cfg or load_config()
    data = build_training_data(cfg)
    model = YieldModel(cfg.model)
    metrics = model.fit(data.X, data.y)
    return model, metrics, data
