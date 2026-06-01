"""End-to-end demo: synthetic weather -> growth simulation -> yield model.

Run from the project root after installing the package:

    python scripts/run_simulation.py

Everything here uses synthetic data so it runs with no external downloads.
Replace ``synthetic_weather`` with real ECCC/NASA POWER ingestion when ready.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canola_dt.config import load_config
from canola_dt.constants import GrowthStage
from canola_dt.data.ingest import synthetic_weather
from canola_dt.features import season_features
from canola_dt.models.yield_model import YieldModel
from canola_dt.simulation.twin import CanolaDigitalTwin


def build_synthetic_training_set(cfg, n_seasons: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic feature/target table to train the demo yield model.

    Yield is a hand-crafted function of GDD, water stress, and flowering heat
    stress (+ noise) so the model has a real signal to learn.
    """
    rng = np.random.default_rng(0)
    rows, targets = [], []
    sim = CanolaDigitalTwin(cfg)

    for i in range(n_seasons):
        weather = synthetic_weather(year=2000 + (i % 24), n_days=130, seed=i)
        feats = season_features(weather)
        feats["sim_final_gdd"] = feats["total_gdd"]
        feats["sim_mean_water_stress"] = min(1.0, feats["max_dry_spell"] / 30.0)
        feats["sim_flowering_heat_days"] = max(0, feats["heat_stress_days"] - 5)

        # Synthetic "true" yield (kg/ha) with agronomically sensible signs.
        y = (
            2600.0
            + 0.15 * (feats["total_gdd"] - 1400)
            + 1.8 * (feats["total_precip_mm"] - 220)
            - 25.0 * feats["sim_flowering_heat_days"]
            - 900.0 * feats["sim_mean_water_stress"]
            + rng.normal(0, 120)
        )
        rows.append(feats)
        targets.append(max(0.0, y))

    X = pd.DataFrame(rows)
    y = pd.Series(targets, name="yield_kg_ha")
    return X, y


def main() -> None:
    cfg = load_config()

    print("== Training synthetic yield model ==")
    X, y = build_synthetic_training_set(cfg)
    model = YieldModel(cfg.model)
    metrics = model.fit(X, y)
    print(f"  validation R^2 : {metrics['r2']:.3f}")
    print(f"  validation MAE : {metrics['mae_kg_ha']:.1f} kg/ha")
    print(f"  train/test     : {metrics['n_train']}/{metrics['n_test']}")

    print("\n== Running digital twin on a held-out synthetic season ==")
    twin = CanolaDigitalTwin(cfg, model=model)
    weather = synthetic_weather(year=2024, n_days=130, seed=999)
    result = twin.run(weather)

    traj = result.state_trajectory
    print(f"  season days        : {len(traj)}")
    print(f"  final cumulative GDD: {traj['cum_gdd'].iloc[-1]:.0f}")
    print(f"  final stage        : {GrowthStage(int(traj['stage'].iloc[-1])).name}")
    print(f"  flowering heat days: {result.features['sim_flowering_heat_days']}")
    print(f"  mean water stress  : {result.features['sim_mean_water_stress']:.2f}")
    print(f"  PREDICTED YIELD    : {result.predicted_yield_kg_ha:.0f} kg/ha")

    print("\n== Crop timing: simulated stage timeline ==")
    for _, r in result.stage_timeline.iterrows():
        print(
            f"  {r['stage_name']:<10} reached {r['date'].date()} "
            f"(day {r['days_from_start']:>3}, {r['cum_gdd']:.0f} GDD)"
        )

    print("\n== Crop timing: in-season forecast (as of day 45) ==")
    as_of = weather["date"].iloc[44]
    inseason = twin.run(weather, as_of=as_of)
    cur = inseason.state_trajectory.iloc[-1]
    print(f"  as of {as_of.date()}: stage {GrowthStage(int(cur['stage'])).name}, {cur['cum_gdd']:.0f} GDD")
    if inseason.stage_forecast.empty:
        print("  (all stages already reached)")
    for _, r in inseason.stage_forecast.iterrows():
        print(
            f"  -> {r['stage_name']:<10} in ~{r['days_until']:>3} days "
            f"({r['forecast_date'].date()}, +{r['gdd_remaining']:.0f} GDD)"
        )


if __name__ == "__main__":
    main()
