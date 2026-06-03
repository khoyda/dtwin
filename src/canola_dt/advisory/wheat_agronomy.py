"""Spring-wheat agronomic constants and enums for the advisory layer.

Sourced from Manitoba/Saskatchewan/Alberta Agriculture, Sask Wheat, the Canadian Grain
Commission and AAFC (provided spec, retrieved 2026-06-01). Drives the wheat alert and
decision-support logic in :mod:`canola_dt.advisory.wheat_engine`.

Reuses :class:`canola_dt.advisory.agronomy.AlertSeverity`. Distinct from the biophysical
:class:`canola_dt.simulation.wheat_model.WheatParameters`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum

LB_PER_AC_TO_KG_PER_HA = 1.12085


class WheatClass(str, Enum):
    CWRS = "CWRS"   # Canada Western Red Spring — bread/milling; protein-driven
    CPSR = "CPSR"   # Canada Prairie Spring Red — softer; larger seed
    CPSW = "CPSW"   # Canada Prairie Spring White


class WheatGrowthStage(IntEnum):
    """Coarse Zadoks-aligned stages (ordered) used for alert timing."""
    GERMINATION = 0    # Z00-09
    EMERGENCE = 1      # Z10
    TILLERING = 2      # Z20s
    JOINTING = 3       # Z30s (stem elongation)
    FLAG_LEAF = 4      # Z37-39  (leaf-disease T1)
    BOOT = 5           # Z45     (midge window opens; strobilurin ban begins)
    HEADING = 6        # Z55     (leaf-disease T2)
    ANTHESIS = 7       # Z60-65  (FHB fungicide window; midge window closes)
    GRAIN_FILL = 8     # Z70-87
    MATURITY = 9       # Z90s
    HARVEST = 10


class WheatPrecedingCrop(str, Enum):
    PULSE = "pulse"          # peas/lentils/fababeans — N benefit, non-host
    CANOLA = "canola"        # oilseed; breaks cereal disease cycle
    FALLOW = "fallow"
    CORN = "corn"
    CEREAL = "cereal"        # barley/oats — cereal disease carryover
    WHEAT = "wheat"          # wheat-on-wheat — worst (FHB, midge, tan spot)
    OTHER = "other"


@dataclass
class WheatAgronomyParameters:
    """Spring-wheat agronomic thresholds for alerts and decision support."""

    wheat_class: WheatClass = WheatClass.CWRS

    # --- Plant establishment (plants per m²) ---
    target_population_per_m2_min: int = 247
    target_population_per_m2_max: int = 301
    thin_stand_per_m2: int = 200          # below -> excess tillering, uneven maturity
    critical_stand_per_m2: int = 120
    expected_stand_loss_pct_min: float = 10.0
    expected_stand_loss_pct_max: float = 20.0
    kernel_weight_g_CWRS: float = 37.0
    kernel_weight_g_CPS: float = 42.0

    # --- Phenology (days after seeding) ---
    days_to_emergence_min: int = 6
    days_to_emergence_max: int = 8
    days_seeding_to_heading_min: int = 55
    days_seeding_to_heading_max: int = 75
    total_season_days_min: int = 85
    total_season_days_max: int = 110
    grain_fill_days_min: int = 13
    grain_fill_days_max: int = 20

    # --- Nitrogen / protein ---
    N_following_stubble_kg_per_ha_min: float = 62.0   # ~55 lb/ac
    N_following_stubble_kg_per_ha_max: float = 101.0  # ~90 lb/ac
    N_min_for_max_yield_CWRS_kg_per_ha: float = 157.0 # ~140 lb/ac
    protein_target_pct: float = 13.5
    protein_max_with_yield_pct: float = 16.0
    post_boot_N_for_protein_kg_per_ha_min: float = 17.0
    post_boot_N_for_protein_kg_per_ha_max: float = 22.0
    # --- P / K / S (kg/ha) ---
    P2O5_recommended_kg_per_ha_min: float = 33.0
    P2O5_recommended_kg_per_ha_max: float = 44.0
    S_recommended_kg_per_ha: float = 17.0
    K2O_sandy_soils_kg_per_ha_min: float = 17.0
    K2O_sandy_soils_kg_per_ha_max: float = 34.0

    # --- Fusarium head blight (FHB) ---
    fhb_fungicide_zadoks_start: int = 60
    fhb_fungicide_zadoks_end: int = 65
    fhb_strobilurin_ban_from_stage: WheatGrowthStage = WheatGrowthStage.BOOT
    fhb_favourable_humidity_pct: float = 70.0
    fhb_favourable_humidity_hrs: float = 36.0
    fhb_favourable_temp_min_c: float = 20.0
    fhb_favourable_temp_max_c: float = 25.0
    fhb_max_days_after_anthesis: int = 7

    # --- Wheat midge (Sitodiplosis mosellana) ---
    midge_window_open_stage: WheatGrowthStage = WheatGrowthStage.BOOT
    midge_window_close_stage: WheatGrowthStage = WheatGrowthStage.ANTHESIS
    midge_threshold_yield_per_head: float = 0.22   # 1 midge / ~4.5 heads
    midge_threshold_grade_per_head: float = 0.11   # 1 midge / ~9 heads
    midge_emergence_precip_min_mm: float = 25.0

    # --- Leaf-disease fungicide timing (Zadoks) ---
    leaf_fungicide_T1_stage: WheatGrowthStage = WheatGrowthStage.FLAG_LEAF
    leaf_fungicide_T2_stage: WheatGrowthStage = WheatGrowthStage.HEADING

    # --- Insect thresholds ---
    aphid_threshold_seedling_per_stem: int = 30   # English grain aphid
    aphid_threshold_boot_per_stem: int = 50
    cutworm_pale_western_per_m2: float = 3.5
    grasshopper_control_required_per_m2: int = 13
    armyworm_per_m2: int = 10

    # --- Lodging / harvest ---
    lodging_yield_loss_pct_min: float = 8.0
    lodging_yield_loss_pct_max: float = 61.0
    swath_kernel_moisture_pct: float = 35.0
    combine_no_dry_moisture_pct: float = 14.0
    combine_with_dry_moisture_pct: float = 20.0
    storage_safe_moisture_pct: float = 14.5

    # --- Rotation ---
    rotation_min_years: int = 2

    # Preceding-crop yield multipliers (relative to cereal stubble baseline).
    preceding_crop_yield_index: dict[str, float] = field(default_factory=lambda: {
        WheatPrecedingCrop.PULSE.value: 1.06,
        WheatPrecedingCrop.FALLOW.value: 1.05,
        WheatPrecedingCrop.CANOLA.value: 1.03,
        WheatPrecedingCrop.CORN.value: 1.00,
        WheatPrecedingCrop.OTHER.value: 0.99,
        WheatPrecedingCrop.CEREAL.value: 0.94,
        WheatPrecedingCrop.WHEAT.value: 0.90,  # wheat-on-wheat: FHB/midge/tan spot
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WheatAgronomyParameters":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
