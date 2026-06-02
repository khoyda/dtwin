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

import pandas as pd

from canola_dt.config import Config, load_config
from canola_dt.data import aafc, eccc, statcan
from canola_dt.models.yield_model import YieldModel
from canola_dt.simulation.twin import CanolaDigitalTwin

MIN_SEASON_COMPLETENESS = 0.80  # drop province-years with too much missing weather


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

def build_training_data(cfg: Config | None = None) -> TrainingData:
    """Join weather features to StatCan (and optional AAFC) yields."""
    cfg = cfg or load_config()
    ds = cfg["data_sources"]
    feats = build_weather_features(cfg)

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
