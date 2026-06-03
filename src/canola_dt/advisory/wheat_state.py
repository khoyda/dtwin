"""Virtual-entity state for the spring-wheat advisory layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date

from canola_dt.advisory.agronomy import AlertSeverity
from canola_dt.advisory.state import Alert
from canola_dt.advisory.wheat_agronomy import (
    WheatClass,
    WheatGrowthStage,
    WheatPrecedingCrop,
)


@dataclass
class WheatFieldState:
    """Virtual entity representing a single spring-wheat field."""

    # Identification
    field_id: str = "WHEAT-001"
    wheat_class: WheatClass = WheatClass.CWRS
    seeding_date: date = field(default_factory=date.today)
    preceding_crop: WheatPrecedingCrop = WheatPrecedingCrop.CANOLA
    years_since_last_wheat: int = 2
    latitude: float = 52.0

    # Agronomic inputs (set at seeding)
    target_population_per_m2: float = 275.0
    n_applied_kg_per_ha: float = 110.0
    p2o5_applied_kg_per_ha: float = 40.0
    s_applied_kg_per_ha: float = 12.0
    k2o_applied_kg_per_ha: float = 0.0
    # Plant-available soil supply (kg/ha) for the nutrient-limited yield ceiling.
    # K defaults high — most prairie soils are K-sufficient (override for sandy/organic).
    soil_available_n_kg_per_ha: float = 30.0
    soil_available_p2o5_kg_per_ha: float = 25.0
    soil_available_k2o_kg_per_ha: float = 300.0
    soil_available_s_kg_per_ha: float = 8.0

    # Current state (updated by sensor readings)
    day_of_season: int = 0
    growth_stage: WheatGrowthStage = WheatGrowthStage.GERMINATION
    plant_population_per_m2: float = 0.0
    air_temp_max_c: float = 20.0
    relative_humidity_pct: float = 50.0       # for FHB risk at flowering
    soil_moisture_pct: float = 25.0
    season_precipitation_mm: float = 0.0
    may_precipitation_mm: float = 0.0          # drives wheat-midge emergence
    midge_per_head: float = 0.0
    aphids_per_stem: float = 0.0
    leaf_disease_severity_pct: float = 0.0
    lodging_pct: float = 0.0
    grain_moisture_pct: float = 100.0

    # Accumulated trackers
    fhb_risk_events: int = 0
    disease_pressure: dict[str, float] = field(default_factory=lambda: {
        "fusarium_head_blight": 0.0,
        "leaf_rust": 0.0,
        "stripe_rust": 0.0,
    })

    # Derived / simulated outputs
    yield_potential_t_ha: float = 0.0
    yield_potential_bu_ac: float = 0.0
    yield_breakdown: dict = field(default_factory=dict)
    estimated_protein_pct: float = 0.0
    estimated_harvest_date: date | None = None
    alert_log: list[Alert] = field(default_factory=list)

    def ingest_sensor_reading(
        self,
        day_of_season: int,
        air_temp_max_c: float,
        precipitation_mm: float,
        plant_population_per_m2: float,
        soil_moisture_pct: float,
        relative_humidity_pct: float = 50.0,
        midge_per_head: float = 0.0,
        aphids_per_stem: float = 0.0,
        leaf_disease_severity_pct: float = 0.0,
        lodging_pct: float = 0.0,
        grain_moisture_pct: float | None = None,
        month: int | None = None,
    ) -> None:
        """Perception layer -> virtual-entity state update."""
        self.day_of_season = day_of_season
        self.air_temp_max_c = air_temp_max_c
        self.season_precipitation_mm += precipitation_mm
        if month == 5:
            self.may_precipitation_mm += precipitation_mm
        self.plant_population_per_m2 = plant_population_per_m2
        self.soil_moisture_pct = soil_moisture_pct
        self.relative_humidity_pct = relative_humidity_pct
        self.midge_per_head = midge_per_head
        self.aphids_per_stem = aphids_per_stem
        self.leaf_disease_severity_pct = leaf_disease_severity_pct
        self.lodging_pct = lodging_pct
        if grain_moisture_pct is not None:
            self.grain_moisture_pct = grain_moisture_pct

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeding_date"] = self.seeding_date.isoformat()
        d["estimated_harvest_date"] = (
            self.estimated_harvest_date.isoformat() if self.estimated_harvest_date else None
        )
        d["wheat_class"] = self.wheat_class.value
        d["preceding_crop"] = self.preceding_crop.value
        d["growth_stage"] = int(self.growth_stage)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WheatFieldState":
        d = dict(d)
        d["seeding_date"] = date.fromisoformat(d["seeding_date"])
        d["estimated_harvest_date"] = (
            date.fromisoformat(d["estimated_harvest_date"]) if d.get("estimated_harvest_date") else None
        )
        d["wheat_class"] = WheatClass(d["wheat_class"])
        d["preceding_crop"] = WheatPrecedingCrop(d["preceding_crop"])
        d["growth_stage"] = WheatGrowthStage(int(d["growth_stage"]))
        d["alert_log"] = [
            Alert(
                severity=AlertSeverity(a["severity"]),
                category=a["category"], message=a["message"],
                recommendation=a["recommendation"], day_of_season=a["day_of_season"],
            )
            for a in d.get("alert_log", [])
        ]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
