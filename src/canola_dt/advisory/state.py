"""Virtual-entity state for the advisory layer (Physical-to-Virtual sync).

``CanolaFieldState`` is the stateful core of the digital twin's advisory layer:
updated by perception-layer sensor readings and queried by the advisory engine.
It is JSON-serialisable (network layer) via ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date

from canola_dt.advisory.agronomy import (
    AlertSeverity,
    CultivarType,
    GrowthStage,
    PrecedingCrop,
    Species,
)


@dataclass
class Alert:
    severity: AlertSeverity
    category: str
    message: str
    recommendation: str
    day_of_season: int

    def __str__(self) -> str:
        return f"[{self.severity.value}] Day {self.day_of_season} | {self.category}: {self.message}"


@dataclass
class CanolaFieldState:
    """Virtual entity representing a single canola field in the digital twin."""

    # Identification
    field_id: str = "FIELD-001"
    species: Species = Species.B_NAPUS
    cultivar_type: CultivarType = CultivarType.HYBRID
    seeding_date: date = field(default_factory=date.today)
    preceding_crop: PrecedingCrop = PrecedingCrop.WHEAT
    years_since_last_canola: int = 3
    latitude: float = 52.0  # site latitude, used by the process-model yield bridge

    # --- Agronomic inputs (set at seeding) ---
    target_plant_density_per_m2: float = 65.0
    seed_depth_cm: float = 1.5
    n_applied_kg_per_ha: float = 150.0
    p2o5_applied_kg_per_ha: float = 40.0
    n_seed_row_kg_per_ha: float = 0.0

    # --- Current state (updated by sensor readings) ---
    day_of_season: int = 0
    growth_stage: GrowthStage = GrowthStage.GS0_GERMINATION
    plant_density_per_m2: float = 0.0
    soil_temp_c: float = 10.0
    air_temp_max_c: float = 20.0
    soil_moisture_pct: float = 20.0
    cumulative_precipitation_mm: float = 0.0
    season_precipitation_mm: float = 0.0
    waterlogged_days_consecutive: int = 0
    leaf_defoliation_pct: float = 0.0
    current_seed_moisture_pct: float = 100.0

    # --- Accumulated stress trackers ---
    heat_stress_events_at_flowering: int = 0
    total_waterlogged_days: int = 0
    disease_pressure: dict[str, float] = field(default_factory=lambda: {
        "sclerotinia": 0.0,
        "blackleg": 0.0,
        "clubroot": 0.0,
    })

    # --- Derived / simulated outputs ---
    yield_potential_t_ha: float = 0.0
    yield_potential_bu_ac: float = 0.0
    yield_breakdown: dict = field(default_factory=dict)  # process yield x mgmt modifiers
    estimated_harvest_date: date | None = None
    alert_log: list[Alert] = field(default_factory=list)

    def ingest_sensor_reading(
        self,
        day_of_season: int,
        soil_temp_c: float,
        air_temp_max_c: float,
        precipitation_mm: float,
        plant_density_per_m2: float,
        soil_moisture_pct: float,
        leaf_defoliation_pct: float = 0.0,
        waterlogged_days_consecutive: int = 0,
        seed_moisture_pct: float | None = None,
    ) -> None:
        """Perception layer -> virtual-entity state update."""
        self.day_of_season = day_of_season
        self.soil_temp_c = soil_temp_c
        self.air_temp_max_c = air_temp_max_c
        self.cumulative_precipitation_mm += precipitation_mm
        self.season_precipitation_mm += precipitation_mm
        self.plant_density_per_m2 = plant_density_per_m2
        self.soil_moisture_pct = soil_moisture_pct
        self.leaf_defoliation_pct = leaf_defoliation_pct
        self.waterlogged_days_consecutive = waterlogged_days_consecutive
        if waterlogged_days_consecutive > 0:
            self.total_waterlogged_days += 1
        if seed_moisture_pct is not None:
            self.current_seed_moisture_pct = seed_moisture_pct

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeding_date"] = self.seeding_date.isoformat()
        d["estimated_harvest_date"] = (
            self.estimated_harvest_date.isoformat() if self.estimated_harvest_date else None
        )
        d["species"] = self.species.value
        d["cultivar_type"] = self.cultivar_type.value
        d["growth_stage"] = self.growth_stage.value
        d["preceding_crop"] = self.preceding_crop.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CanolaFieldState":
        d = dict(d)
        d["seeding_date"] = date.fromisoformat(d["seeding_date"])
        d["estimated_harvest_date"] = (
            date.fromisoformat(d["estimated_harvest_date"]) if d.get("estimated_harvest_date") else None
        )
        d["species"] = Species(d["species"])
        d["cultivar_type"] = CultivarType(d["cultivar_type"])
        d["growth_stage"] = GrowthStage(d["growth_stage"])
        d["preceding_crop"] = PrecedingCrop(d["preceding_crop"])
        d["alert_log"] = [
            Alert(
                severity=AlertSeverity(a["severity"]),
                category=a["category"],
                message=a["message"],
                recommendation=a["recommendation"],
                day_of_season=a["day_of_season"],
            )
            for a in d.get("alert_log", [])
        ]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
