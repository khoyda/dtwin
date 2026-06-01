"""CanolaDigitalTwin — orchestrates simulation + yield prediction for a field-season.

The twin holds the virtual representation of one field-season: it runs the
process-based growth model to derive an interpretable state trajectory, extracts
season features, and (optionally) calls the trained ML model to predict yield.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from canola_dt.config import Config, load_config
from canola_dt.data.preprocess import clean_weather
from canola_dt.features import season_features
from canola_dt.models.yield_model import YieldModel
from canola_dt.simulation.growth import GrowthSimulator
from canola_dt.simulation.phenology import (
    expected_daily_gdd,
    forecast_stage_dates,
    stage_timeline,
)


@dataclass
class TwinResult:
    """Bundle of everything the twin produced for a field-season."""

    state_trajectory: pd.DataFrame  # daily DailyState rows
    features: dict[str, float]
    stage_timeline: pd.DataFrame  # crop timing: when each stage was reached
    stage_forecast: pd.DataFrame  # crop timing: predicted dates for upcoming stages
    predicted_yield_kg_ha: float | None = None


class CanolaDigitalTwin:
    """Virtual twin of a single canola field-season."""

    def __init__(self, config: Config | None = None, model: YieldModel | None = None):
        self.config = config or load_config()
        self.simulator = GrowthSimulator.from_config(self.config)
        self.model = model

    def run(self, weather: pd.DataFrame, as_of: pd.Timestamp | None = None) -> TwinResult:
        """Run the full pipeline on a daily weather frame for one field-season.

        ``as_of`` controls the crop-timing forecast origin. If given, the trajectory
        is truncated at that date and upcoming stage dates are forecast forward from
        the observed GDD accrual rate (in-season "where are we / what's next" use).
        If ``None``, the full season is simulated and the forecast covers any stages
        not reached by season end.
        """
        weather = clean_weather(weather)
        trajectory = self.simulator.run(weather)

        if as_of is not None:
            trajectory = trajectory[trajectory["date"] <= as_of].reset_index(drop=True)
            if trajectory.empty:
                raise ValueError("as_of precedes the start of the weather record")

        # --- Crop timing (phenology) ---
        timeline = stage_timeline(trajectory)
        current = trajectory.iloc[-1]
        forecast = forecast_stage_dates(
            current_cum_gdd=float(current["cum_gdd"]),
            current_date=current["date"],
            stage_thresholds=self.simulator.stage_thresholds,
            daily_gdd_rate=expected_daily_gdd(trajectory),
        )

        feats = season_features(
            weather,
            base_temp_c=self.simulator.base_temp_c,
            cap_temp_c=self.simulator.cap_temp_c,
            heat_threshold_c=self.simulator.heat_threshold_c,
        )
        # Surface a couple of simulated-state summaries as features too.
        feats["sim_final_gdd"] = float(trajectory["cum_gdd"].iloc[-1])
        feats["sim_mean_water_stress"] = float(trajectory["water_stress_index"].mean())
        feats["sim_flowering_heat_days"] = int(trajectory["flowering_heat_stress"].sum())

        predicted = None
        if self.model is not None:
            X = pd.DataFrame([feats])
            predicted = float(self.model.predict(X).iloc[0])

        return TwinResult(
            state_trajectory=trajectory,
            features=feats,
            stage_timeline=timeline,
            stage_forecast=forecast,
            predicted_yield_kg_ha=predicted,
        )
