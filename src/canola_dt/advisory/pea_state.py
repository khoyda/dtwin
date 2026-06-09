"""Virtual-entity state for the yellow-pea advisory layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date

from canola_dt.advisory.agronomy import AlertSeverity
from canola_dt.advisory.state import Alert
from canola_dt.advisory.pea_agronomy import PeaGrowthStage, PeaPrecedingCrop, PeaType


@dataclass
class PeaFieldState:
    """Virtual entity representing a single yellow-pea field."""

    field_id: str = "PEA-001"
    pea_type: PeaType = PeaType.YELLOW
    seeding_date: date = field(default_factory=date.today)
    preceding_crop: PeaPrecedingCrop = PeaPrecedingCrop.CEREAL
    years_since_last_pulse: int = 4
    latitude: float = 52.0
    inoculant_applied: bool = True

    # Agronomic inputs (set at seeding)
    target_population_per_m2: float = 80.0
    n_applied_kg_per_ha: float = 12.0       # starter only — peas fix their N
    p2o5_applied_kg_per_ha: float = 40.0
    s_applied_kg_per_ha: float = 8.0
    k2o_applied_kg_per_ha: float = 0.0
    soil_available_n_kg_per_ha: float = 25.0
    soil_available_p2o5_kg_per_ha: float = 20.0
    soil_available_k2o_kg_per_ha: float = 300.0
    soil_available_s_kg_per_ha: float = 8.0

    # Current state (updated by sensor readings)
    day_of_season: int = 0
    growth_stage: PeaGrowthStage = PeaGrowthStage.GERMINATION
    plant_population_per_m2: float = 0.0
    air_temp_max_c: float = 20.0
    soil_moisture_pct: float = 25.0
    season_precipitation_mm: float = 0.0
    aphids_per_tip: float = 0.0
    weevil_damage_pct: float = 0.0
    ascochyta_severity_pct: float = 0.0
    cutworm_larvae_per_m2: float = 0.0
    lodging_pct: float = 0.0
    pod_brown_pct: float = 0.0
    grain_moisture_pct: float = 100.0

    disease_pressure: dict[str, float] = field(default_factory=lambda: {
        "ascochyta": 0.0, "aphanomyces": 0.0,
    })

    # Derived / simulated outputs
    yield_potential_t_ha: float = 0.0
    yield_potential_bu_ac: float = 0.0
    yield_breakdown: dict = field(default_factory=dict)
    estimated_protein_pct: float = 0.0
    estimated_harvest_date: date | None = None
    alert_log: list[Alert] = field(default_factory=list)

    def ingest_sensor_reading(
        self, day_of_season: int, air_temp_max_c: float, precipitation_mm: float,
        plant_population_per_m2: float, soil_moisture_pct: float,
        aphids_per_tip: float = 0.0, weevil_damage_pct: float = 0.0,
        ascochyta_severity_pct: float = 0.0, cutworm_larvae_per_m2: float = 0.0,
        lodging_pct: float = 0.0, pod_brown_pct: float = 0.0,
        grain_moisture_pct: float | None = None,
    ) -> None:
        self.day_of_season = day_of_season
        self.air_temp_max_c = air_temp_max_c
        self.season_precipitation_mm += precipitation_mm
        self.plant_population_per_m2 = plant_population_per_m2
        self.soil_moisture_pct = soil_moisture_pct
        self.aphids_per_tip = aphids_per_tip
        self.weevil_damage_pct = weevil_damage_pct
        self.ascochyta_severity_pct = ascochyta_severity_pct
        self.cutworm_larvae_per_m2 = cutworm_larvae_per_m2
        self.lodging_pct = lodging_pct
        self.pod_brown_pct = pod_brown_pct
        if grain_moisture_pct is not None:
            self.grain_moisture_pct = grain_moisture_pct

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeding_date"] = self.seeding_date.isoformat()
        d["estimated_harvest_date"] = (
            self.estimated_harvest_date.isoformat() if self.estimated_harvest_date else None)
        d["pea_type"] = self.pea_type.value
        d["preceding_crop"] = self.preceding_crop.value
        d["growth_stage"] = int(self.growth_stage)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PeaFieldState":
        d = dict(d)
        d["seeding_date"] = date.fromisoformat(d["seeding_date"])
        d["estimated_harvest_date"] = (
            date.fromisoformat(d["estimated_harvest_date"]) if d.get("estimated_harvest_date") else None)
        d["pea_type"] = PeaType(d["pea_type"])
        d["preceding_crop"] = PeaPrecedingCrop(d["preceding_crop"])
        d["growth_stage"] = PeaGrowthStage(int(d["growth_stage"]))
        d["alert_log"] = [
            Alert(severity=AlertSeverity(a["severity"]), category=a["category"],
                  message=a["message"], recommendation=a["recommendation"],
                  day_of_season=a["day_of_season"])
            for a in d.get("alert_log", [])
        ]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
