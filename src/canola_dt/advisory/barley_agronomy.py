"""Spring-barley agronomic constants and enums for the advisory layer.

Sourced from Manitoba/Saskatchewan/Alberta Agriculture, AAFC Lacombe, the barley
development commissions and crop-science extension (provided spec, retrieved 2026-06-01).
Drives the barley alert/decision-support logic in :mod:`canola_dt.advisory.barley_engine`.

Reuses :class:`canola_dt.advisory.agronomy.AlertSeverity`. Distinct from the biophysical
:class:`canola_dt.simulation.barley_model.BarleyParameters`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum


class BarleyType(str, Enum):
    MALT_2ROW = "malt_2row"   # primary malting class; protein-constrained
    MALT_6ROW = "malt_6row"
    FEED = "feed"             # no protein ceiling; tolerates higher N


class BarleyGrowthStage(IntEnum):
    """Coarse Zadoks-aligned stages (ordered) used for alert timing."""
    GERMINATION = 0    # Z00-09
    EMERGENCE = 1      # Z10
    TILLERING = 2      # Z20s   (herbicide window)
    JOINTING = 3       # Z30-33 (PGR window; scald critical)
    FLAG_LEAF = 4      # Z37-39 (net blotch / scald T2 — key leaf)
    BOOT = 5           # Z45    (FHB window opens; strobilurin ban)
    HEADING = 6        # Z55
    ANTHESIS = 7       # Z60-65 (primary FHB fungicide)
    GRAIN_FILL = 8     # Z70-87
    MATURITY = 9       # Z90s
    HARVEST = 10


class BarleyPrecedingCrop(str, Enum):
    SUMMERFALLOW = "summerfallow"
    PULSE = "pulse"          # peas / lentils
    CANOLA = "canola"
    CORN = "corn"
    CEREAL = "cereal"        # wheat / oats
    BARLEY = "barley"        # barley-on-barley — worst (net blotch, scald, root rot)
    OTHER = "other"


@dataclass
class BarleyAgronomyParameters:
    """Spring-barley agronomic thresholds for alerts and decision support."""

    barley_type: BarleyType = BarleyType.MALT_2ROW

    # --- Plant establishment (plants per m²) ---
    malt_target_population_per_m2_min: int = 195
    malt_target_population_per_m2_max: int = 270
    malt_optimum_dryland_per_m2: int = 200
    feed_target_population_per_m2_min: int = 237
    feed_target_population_per_m2_max: int = 269
    thin_stand_per_m2: int = 150
    critical_stand_per_m2: int = 100

    # --- Phenology (days after seeding) ---
    days_to_emergence_min: int = 6
    days_to_emergence_max: int = 8
    days_seeding_to_heading_min: int = 50
    days_seeding_to_heading_max: int = 70
    total_season_days_min: int = 80
    total_season_days_max: int = 105

    # --- Nitrogen / protein ---
    N_following_stubble_kg_per_ha_min: float = 62.0
    N_following_stubble_kg_per_ha_max: float = 101.0
    N_malt_max_safe_kg_per_ha: float = 100.0   # stay <=12.5% protein (modern varieties)
    N_feed_high_yield_kg_per_ha: float = 140.0
    malt_protein_min_pct: float = 11.0
    malt_protein_max_pct: float = 12.5

    # --- P / K / S (kg/ha) ---
    P2O5_recommended_kg_per_ha_min: float = 33.6
    P2O5_recommended_kg_per_ha_max: float = 44.8
    S_recommended_kg_per_ha: float = 16.8
    K2O_sandy_soils_kg_per_ha_min: float = 17.0
    K2O_sandy_soils_kg_per_ha_max: float = 34.0

    # --- Leaf disease: net blotch & scald ---
    net_blotch_fungicide_zadoks_start: int = 39   # flag leaf
    net_blotch_fungicide_zadoks_end: int = 59
    net_blotch_severity_threshold_pct: float = 5.0   # visible on upper 3 leaves
    scald_economic_threshold_pct: float = 2.0        # 1-2% at GS31-32
    excess_N_net_blotch_note: bool = True

    # --- Fusarium head blight (FHB) ---
    fhb_fungicide_zadoks_start: int = 60
    fhb_fungicide_zadoks_end: int = 65
    fhb_strobilurin_ban_from_stage: BarleyGrowthStage = BarleyGrowthStage.BOOT
    fhb_favourable_humidity_pct: float = 70.0
    fhb_favourable_temp_min_c: float = 20.0
    fhb_favourable_temp_max_c: float = 25.0

    # --- Lodging / PGR ---
    lodging_yield_loss_pct_min: float = 8.0
    lodging_yield_loss_pct_max: float = 30.0
    pgr_moddus_zadoks_min: int = 30
    pgr_moddus_zadoks_max: int = 33
    pgr_high_yield_target_bu_ac: float = 80.0     # PGR only justified in high-yield env

    # --- Insects ---
    cutworm_pale_western_per_m2: float = 3.5
    grasshopper_control_required_per_m2: int = 13
    aphid_threshold_seedling_per_stem: int = 30

    # --- Harvest ---
    swath_kernel_moisture_pct: float = 30.0
    combine_with_dry_moisture_min_pct: float = 18.0
    combine_with_dry_moisture_max_pct: float = 20.0
    combine_straight_cut_start_pct: float = 16.0
    malt_combine_floor_moisture_pct: float = 13.5   # below this peels kernels -> quality loss
    malt_storage_moisture_max_pct: float = 13.5
    feed_storage_moisture_max_pct: float = 14.5

    # --- Rotation ---
    rotation_min_years: int = 2

    preceding_crop_yield_index: dict[str, float] = field(default_factory=lambda: {
        BarleyPrecedingCrop.SUMMERFALLOW.value: 1.10,
        BarleyPrecedingCrop.PULSE.value: 1.07,
        BarleyPrecedingCrop.CANOLA.value: 1.05,
        BarleyPrecedingCrop.CORN.value: 1.00,
        BarleyPrecedingCrop.OTHER.value: 0.99,
        BarleyPrecedingCrop.CEREAL.value: 0.93,
        BarleyPrecedingCrop.BARLEY.value: 0.88,   # barley-on-barley
    })

    def is_malt(self) -> bool:
        return self.barley_type in (BarleyType.MALT_2ROW, BarleyType.MALT_6ROW)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BarleyAgronomyParameters":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
