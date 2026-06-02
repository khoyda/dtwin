"""Agronomic constants and enums for the advisory (decision-support) layer.

Sourced from the Canola Council of Canada Canola Encyclopedia
(https://www.canolacouncil.org/canola-encyclopedia/, retrieved 2026-05-31).

These threshold/management constants drive the alert and decision-support logic in
:mod:`canola_dt.advisory.engine`. They are distinct from the *biophysical* crop
parameters in :class:`canola_dt.simulation.process_model.CanolaParameters` (which is
why this class is named ``AgronomyParameters`` — the two must not be confused).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Species(str, Enum):
    """Three canola species grown in western Canada."""
    B_NAPUS = "B_napus"      # Argentine — most common, highest yield
    B_RAPA = "B_rapa"        # Polish — earlier maturity, more shatter-resistant
    B_JUNCEA = "B_juncea"    # Canola-quality brown mustard — dry/hot southern prairies


class CultivarType(str, Enum):
    HYBRID = "hybrid"                        # Higher yield; needs +50-60 kg N/ha extra
    OPEN_POLLINATED = "open_pollinated"
    SHATTER_RESISTANT = "shatter_resistant"  # Preferred for straight-cut harvest


class GrowthStage(str, Enum):
    """BBCH decimal system — stages relevant to canola DT decision-support."""
    GS0_GERMINATION = "GS0_Germination"
    GS1_LEAF_DEV = "GS1_LeafDevelopment"
    GS3_STEM_ELONG = "GS3_StemElongation"
    GS5_FLOWERING = "GS5_Flowering"
    GS7_POD_FILL = "GS7_PodFill"
    GS8_SEED_FILL = "GS8_SeedFill"
    GS9_MATURITY = "GS9_Maturity"
    HARVESTED = "Harvested"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PrecedingCrop(str, Enum):
    """Ranked by canola yield response (Canola Council, cropping sequence)."""
    FABABEAN_GREEN_MANURE = "fababean_green_manure"  # Rank 1 — highest yield
    PEAS = "peas"                                    # Rank 2
    LENTILS = "lentils"                              # Rank 3
    FABABEAN_HARVESTED = "fababean_harvested"        # Rank 4
    WHEAT = "wheat"                                  # Rank 5 (baseline)
    CANOLA = "canola"                                # Avoid — disease risk
    OTHER_CEREAL = "other_cereal"


@dataclass
class AgronomyParameters:
    """Canola Council agronomic thresholds for alerts and decision support."""

    # --- Species / variety ---
    default_species: Species = Species.B_NAPUS

    # --- Plant establishment ---
    plant_density_optimal_min: int = 50      # plants / m²
    plant_density_optimal_max: int = 80
    plant_density_warning_min: int = 30      # yield drops below here
    plant_density_warning_max: int = 40
    plant_density_critical_min: int = 10     # high crop failure risk
    plant_density_critical_max: int = 20
    plant_density_lodging_risk: int = 200    # above this: lodging risk
    seed_survivability_pct_min: float = 50.0
    seed_survivability_pct_max: float = 60.0
    seed_depth_optimal_min_cm: float = 1.0
    seed_depth_optimal_max_cm: float = 2.0
    seed_depth_range_min_cm: float = 1.25
    seed_depth_range_max_cm: float = 3.0

    # --- Growth stage timing (days after seeding) ---
    days_to_emergence_min: int = 4
    days_to_emergence_max: int = 15
    days_to_first_flower_min: int = 40
    days_to_first_flower_max: int = 60
    days_flower_to_seed_fill_min: int = 35
    days_flower_to_seed_fill_max: int = 45
    phys_maturity_seed_moisture_pct_min: float = 30.0
    phys_maturity_seed_moisture_pct_max: float = 35.0

    # --- Plant morphology ---
    plant_height_cm_min: float = 75.0
    plant_height_cm_max: float = 175.0
    root_depth_at_maturity_cm_avg: float = 140.0
    root_growth_rate_cm_per_day: float = 2.0
    LAI_for_90pct_solar_interception: float = 4.0
    branches_normal_density_min: int = 3
    branches_normal_density_max: int = 5
    branches_low_density_multiplier: float = 4.0
    low_density_maturity_delay_max_days: int = 21
    secondary_branch_pct_yield_at_20_plants: float = 80.0

    # --- Harvest triggers ---
    swath_colour_change_pct_min: float = 30.0   # % on main stem (normal stand)
    swath_colour_change_pct_max: float = 40.0
    swath_colour_change_thin_stand: float = 60.0
    straight_cut_max_yield_loss_pct: float = 5.0
    seed_moisture_loss_rate_pct_per_day_min: float = 2.0
    seed_moisture_loss_rate_pct_per_day_max: float = 3.0

    # --- Environmental stress ---
    heat_stress_flowering_threshold_c: float = 29.5
    heat_yield_loss_t_ha_per_3c_increase: float = 0.4  # Nuttall et al. 1992
    heat_reference_temp_c: float = 21.0
    waterlogging_trigger_days: int = 3
    yield_per_mm_kg_min: float = 1.8
    yield_per_mm_kg_max: float = 3.3

    # --- Nutrient management ---
    hybrid_extra_N_kg_per_ha_min: float = 50.0
    hybrid_extra_N_kg_per_ha_max: float = 60.0
    N_seed_row_risk_threshold_kg_per_ha: float = 101.0
    P_seed_row_safe_max_dry_kg_per_ha: float = 22.0
    P_seed_row_safe_max_moist_kg_per_ha: float = 28.0
    N_inhibits_P_uptake_above_kg_per_ha: float = 90.0
    N_in_seed_pct_min: float = 3.4
    N_in_seed_pct_max: float = 4.0
    N_uptake_peak_weeks_after_emergence_min: int = 9
    N_uptake_peak_weeks_after_emergence_max: int = 10

    # --- Pest / disease economic thresholds ---
    flea_beetle_action_threshold_pct: float = 25.0
    flea_beetle_tolerable_loss_pct: float = 50.0
    cutworm_threshold_min_per_m2: float = 4.0
    cutworm_threshold_max_per_m2: float = 6.0
    lygus_standard_threshold_per_10_sweeps: int = 5
    sclerotinia_max_yield_loss_pct: float = 33.0
    sclerotinia_fungicide_trigger_pct_flower: float = 10.0

    # --- Cropping sequence / rotation ---
    rotation_min_years: int = 2
    rotation_recommended_years: int = 4
    canola_stubble_degradation_multiplier: float = 2.0

    # --- Preceding crop yield multipliers (relative to wheat = 1.0) ---
    preceding_crop_yield_index: dict[str, float] = field(default_factory=lambda: {
        PrecedingCrop.FABABEAN_GREEN_MANURE.value: 1.12,
        PrecedingCrop.PEAS.value: 1.08,
        PrecedingCrop.LENTILS.value: 1.06,
        PrecedingCrop.FABABEAN_HARVESTED.value: 1.05,
        PrecedingCrop.WHEAT.value: 1.00,
        PrecedingCrop.OTHER_CEREAL.value: 0.98,
        PrecedingCrop.CANOLA.value: 0.85,  # disease risk
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgronomyParameters":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
