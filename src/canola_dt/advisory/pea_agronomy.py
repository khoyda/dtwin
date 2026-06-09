"""Yellow-pea agronomic constants and enums for the advisory layer.

Sourced from Saskatchewan Pulse Growers, Government of Saskatchewan/Manitoba/Alberta
Agriculture and AAFC (provided spec, retrieved 2026-06-07). Drives the pea alert logic in
:mod:`canola_dt.advisory.pea_engine`.

Yellow pea is a cool-season **legume**: it fixes most of its own N (rhizobia), so high N is
*harmful* (it suppresses nodulation). It is very heat-sensitive (flowers abort above ~25 C)
and aphanomyces root rot makes a long pulse-free rotation essential.

Reuses :class:`canola_dt.advisory.agronomy.AlertSeverity`. Distinct from the biophysical
:class:`canola_dt.simulation.pea_model.PeaParameters`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum


class PeaType(str, Enum):
    YELLOW = "yellow"   # yellow cotyledon field pea (most common)
    GREEN = "green"


class PeaGrowthStage(IntEnum):
    """Coarse stages (ordered) used for alert timing."""
    GERMINATION = 0
    EMERGENCE = 1
    VEGETATIVE = 2     # node stages; weevil + herbicide windows
    FLOWERING = 3      # heat-abort + ascochyta fungicide window
    POD_FILL = 4
    MATURITY = 5
    HARVEST = 6


class PeaPrecedingCrop(str, Enum):
    CEREAL = "cereal"        # wheat / barley / oats — ideal before peas
    CANOLA = "canola"
    FALLOW = "fallow"
    OTHER = "other"
    PULSE = "pulse"          # lentil / faba / chickpea — disease carryover
    PEA = "pea"              # pea-on-pea — worst (aphanomyces, root rot)


@dataclass
class PeaAgronomyParameters:
    """Yellow-pea agronomic thresholds for alerts and decision support."""

    pea_type: PeaType = PeaType.YELLOW

    # --- Plant establishment (plants per m²) ---
    target_population_per_m2_min: int = 75
    target_population_per_m2_max: int = 85
    thin_stand_per_m2: int = 55
    critical_stand_per_m2: int = 35
    expected_emergence_pct: float = 88.0
    tkw_g: float = 235.0       # CDC Amarillo example

    # --- Phenology (days after seeding) ---
    days_to_emergence_min: int = 7
    days_to_emergence_max: int = 10
    days_seeding_to_flowering_min: int = 40
    days_seeding_to_flowering_max: int = 55
    total_season_days_min: int = 85
    total_season_days_max: int = 100
    heat_abort_threshold_c: float = 25.0

    # --- Nitrogen fixation / fertility ---
    N_fixation_kg_per_ha_min: float = 40.0
    N_fixation_kg_per_ha_max: float = 80.0
    starter_N_kg_per_ha_min: float = 10.0
    starter_N_kg_per_ha_max: float = 15.0
    N_nodulation_inhibit_kg_per_ha: float = 28.0     # N above this suppresses nodulation
    N_fixation_prevented_kg_per_ha: float = 55.0     # N above this prevents fixation
    N_credit_to_next_crop_kg_per_ha_min: float = 20.0
    N_credit_to_next_crop_kg_per_ha_max: float = 40.0
    inoculant_required: bool = True

    # --- Phosphorus / S (kg/ha) ---
    P2O5_optimal_kg_per_ha_min: float = 39.0
    P2O5_optimal_kg_per_ha_max: float = 50.0
    P2O5_seed_row_safe_max_kg_per_ha: float = 20.0
    P2O5_response_olsen_threshold_ppm: float = 15.0
    S_recommended_kg_per_ha: float = 10.0

    # --- Insects ---
    pea_aphid_threshold_per_tip: float = 3.0           # per 20 cm tip
    pea_leaf_weevil_damage_threshold_pct: float = 30.0  # % plants with feeding notches
    pea_leaf_weevil_window_node_start: int = 2
    pea_leaf_weevil_window_node_end: int = 5
    cutworm_per_m2: float = 3.5
    grasshopper_control_required_per_m2: int = 13

    # --- Diseases ---
    ascochyta_yield_loss_max_pct: float = 80.0
    ascochyta_fungicide_timing: str = "early flowering (+10-14 days if wet)"
    aphanomyces_soil_persistence_years_min: int = 10
    aphanomyces_soil_persistence_years_max: int = 15

    # --- Harvest ---
    physiological_maturity_moisture_pct: float = 35.0
    swath_bottom_pods_ripe_pct: float = 30.0
    desiccation_pod_brown_pct: float = 75.0
    desiccation_moisture_pct: float = 30.0
    harvest_safe_moisture_pct: float = 16.0
    storage_safe_moisture_pct: float = 16.0

    # --- Quality ---
    protein_pct_min: float = 22.0
    protein_pct_max: float = 25.0
    protein_premium_threshold_pct: float = 23.5

    # --- Rotation (aphanomyces) ---
    rotation_min_years_from_pulse: int = 3
    rotation_recommended_years_from_pulse: int = 4

    preceding_crop_yield_index: dict[str, float] = field(default_factory=lambda: {
        PeaPrecedingCrop.CEREAL.value: 1.05,   # peas do best after a cereal
        PeaPrecedingCrop.CANOLA.value: 1.02,
        PeaPrecedingCrop.FALLOW.value: 1.00,
        PeaPrecedingCrop.OTHER.value: 1.00,
        PeaPrecedingCrop.PULSE.value: 0.92,    # disease carryover
        PeaPrecedingCrop.PEA.value: 0.85,      # pea-on-pea (aphanomyces/root rot)
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PeaAgronomyParameters":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
