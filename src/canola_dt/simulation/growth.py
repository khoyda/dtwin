"""Process-based canola growth model: phenology + simple water balance.

This is a deliberately lightweight bucket model — enough to give the twin an
interpretable internal state that evolves day by day and can later be corrected by
data assimilation. It is *not* a replacement for APSIM/DSSAT-grade crop models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from canola_dt import constants
from canola_dt.constants import GrowthStage
from canola_dt.features import growing_degree_days


@dataclass
class DailyState:
    """Simulated crop/soil state on a single day."""

    date: pd.Timestamp
    cum_gdd: float
    stage: GrowthStage
    soil_water_mm: float
    water_stress_index: float  # 0 (no stress) .. 1 (fully stressed)
    flowering_heat_stress: bool


@dataclass
class GrowthSimulator:
    """Step a canola crop forward over a daily weather frame.

    Parameters mirror ``config.yaml`` (agronomy + water_balance sections).
    """

    stage_thresholds: dict[GrowthStage, float] = field(
        default_factory=lambda: dict(constants.DEFAULT_STAGE_GDD_THRESHOLDS)
    )
    base_temp_c: float = constants.GDD_BASE_TEMP_C
    cap_temp_c: float = constants.GDD_CAP_TEMP_C
    heat_threshold_c: float = constants.HEAT_STRESS_THRESHOLD_C
    soil_water_capacity_mm: float = 150.0
    initial_soil_water_frac: float = 0.7
    crop_coefficient_kc: float = 1.0

    @classmethod
    def from_config(cls, cfg) -> "GrowthSimulator":
        agro = cfg.agronomy
        wb = cfg.water_balance
        thresholds = {
            GrowthStage[name.upper()]: float(val)
            for name, val in agro["stage_gdd_thresholds"].items()
        }
        return cls(
            stage_thresholds=thresholds,
            base_temp_c=float(agro["gdd_base_temp_c"]),
            cap_temp_c=float(agro["gdd_cap_temp_c"]),
            heat_threshold_c=float(agro["heat_stress_threshold_c"]),
            soil_water_capacity_mm=float(wb["soil_water_capacity_mm"]),
            initial_soil_water_frac=float(wb["initial_soil_water_frac"]),
            crop_coefficient_kc=float(wb["crop_coefficient_kc"]),
        )

    def _stage_for_gdd(self, cum_gdd: float) -> GrowthStage:
        stage = GrowthStage.EMERGENCE if cum_gdd > 0 else GrowthStage.SOWN
        for s in sorted(self.stage_thresholds, key=lambda k: self.stage_thresholds[k]):
            if cum_gdd >= self.stage_thresholds[s]:
                stage = s
        return stage

    def _reference_et_mm(self, tmean_c: float) -> float:
        """Crude temperature-based reference ET (Hargreaves-lite proxy)."""
        return max(0.0, 0.0023 * (tmean_c + 17.8) * 5.0)

    def run(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Simulate the season; returns one row of :class:`DailyState` per day."""
        gdd_daily = growing_degree_days(weather["tmean_c"], self.base_temp_c, self.cap_temp_c)
        soil_water = self.soil_water_capacity_mm * self.initial_soil_water_frac
        cum_gdd = 0.0
        records: list[DailyState] = []

        for i, row in weather.reset_index(drop=True).iterrows():
            cum_gdd += float(gdd_daily.iloc[i])
            stage = self._stage_for_gdd(cum_gdd)

            # Water balance: + rain, - crop ET, clamped to [0, capacity].
            et = self._reference_et_mm(float(row["tmean_c"])) * self.crop_coefficient_kc
            soil_water = min(
                self.soil_water_capacity_mm,
                max(0.0, soil_water + float(row["precip_mm"]) - et),
            )
            stress = 1.0 - (soil_water / self.soil_water_capacity_mm)

            heat = (
                stage == GrowthStage.FLOWERING
                and float(row["tmax_c"]) > self.heat_threshold_c
            )

            records.append(
                DailyState(
                    date=row["date"],
                    cum_gdd=round(cum_gdd, 2),
                    stage=stage,
                    soil_water_mm=round(soil_water, 2),
                    water_stress_index=round(stress, 3),
                    flowering_heat_stress=heat,
                )
            )

        return pd.DataFrame(records)
