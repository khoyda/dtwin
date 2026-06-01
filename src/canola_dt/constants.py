"""Agronomic constants for canola (*Brassica napus*) on the Canadian Prairies.

Values are conventional defaults drawn from Canola Council of Canada guidance and
GDD-based phenology literature. They are intended as calibration starting points,
not validated parameters — tune against regional data before operational use.
"""

from __future__ import annotations

from enum import IntEnum

# Growing-degree-day parameters (base temperature in deg C).
GDD_BASE_TEMP_C: float = 5.0
GDD_CAP_TEMP_C: float = 30.0

# Reproductive heat stress: daily max temperature (deg C) above which canola
# flowers abort, sharply reducing pod/seed set if sustained during flowering.
HEAT_STRESS_THRESHOLD_C: float = 29.5


class GrowthStage(IntEnum):
    """Ordered canola phenological stages (coarse, GDD-driven)."""

    SOWN = 0
    EMERGENCE = 1
    ROSETTE = 2
    BOLTING = 3
    FLOWERING = 4
    RIPENING = 5
    MATURITY = 6


# Cumulative GDD (base 5 C) at which each stage is reached. Defaults; override
# via config.yaml -> agronomy.stage_gdd_thresholds.
DEFAULT_STAGE_GDD_THRESHOLDS: dict[GrowthStage, float] = {
    GrowthStage.EMERGENCE: 120.0,
    GrowthStage.ROSETTE: 350.0,
    GrowthStage.BOLTING: 600.0,
    GrowthStage.FLOWERING: 900.0,
    GrowthStage.RIPENING: 1500.0,
    GrowthStage.MATURITY: 1900.0,
}

# Typical Prairie target plant stand and yield range for sanity checks.
TARGET_PLANT_STAND_PER_M2: tuple[int, int] = (50, 80)
TYPICAL_YIELD_RANGE_KG_HA: tuple[float, float] = (1500.0, 3500.0)
