"""Advisory-layer demo: agronomic alerts + calibrated process-model yield.

    python scripts/run_advisory.py

Runs the decision-support engine over a season of (synthetic) sensor readings to
generate alerts, and sets the field's yield from the CALIBRATED biophysical process
model driven by real ECCC weather — the two halves of the digital twin together.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from canola_dt.advisory import (
    CanolaAdvisoryEngine,
    CanolaFieldState,
    CultivarType,
    PrecedingCrop,
    Species,
    calculate_seeding_rate,
    estimate_n_requirement,
    get_harvest_strategy,
)
from canola_dt.config import load_config
from canola_dt.data import eccc


def _season_weather(cfg, year=2020):
    """A real ECCC season frame (first SK station) for the process-model yield."""
    stations = eccc.station_map(cfg)
    sid, info = next((s, i) for s, i in stations.items() if i["province"] == "Saskatchewan")
    daily = eccc.fetch_daily(int(sid), year, cfg.path("data_raw") / "eccc")
    s = daily[(daily["date"] >= pd.Timestamp(year, 5, 1))
              & (daily["date"] <= pd.Timestamp(year, 10, 31))].set_index("date").asfreq("D")
    s[["tmin_c", "tmax_c", "tmean_c"]] = s[["tmin_c", "tmax_c", "tmean_c"]].interpolate(limit=5)
    s["precip_mm"] = s["precip_mm"].fillna(0.0)
    return s.reset_index(), float(info["lat"])


def main() -> None:
    print("=" * 65)
    print("  Canola Digital Twin — Advisory layer + calibrated yield")
    print("=" * 65)

    cfg = load_config()
    engine = CanolaAdvisoryEngine.with_calibrated_model(cfg)
    weather, lat = _season_weather(cfg, 2020)

    state = CanolaFieldState(
        field_id="SK-2020-DEMO",
        species=Species.B_NAPUS,
        cultivar_type=CultivarType.HYBRID,
        seeding_date=date(2020, 5, 5),
        preceding_crop=PrecedingCrop.PEAS,
        years_since_last_canola=3,
        latitude=lat,
        n_applied_kg_per_ha=160.0,
        p2o5_applied_kg_per_ha=40.0,
        n_seed_row_kg_per_ha=10.0,
    )

    sensor_log = [
        {"day_of_season": 7, "soil_temp_c": 10.0, "air_temp_max_c": 16.0, "precipitation_mm": 12,
         "plant_density_per_m2": 0, "soil_moisture_pct": 28, "waterlogged_days_consecutive": 0},
        {"day_of_season": 14, "soil_temp_c": 12.5, "air_temp_max_c": 19.0, "precipitation_mm": 8,
         "plant_density_per_m2": 58, "soil_moisture_pct": 26, "waterlogged_days_consecutive": 0},
        {"day_of_season": 21, "soil_temp_c": 14.0, "air_temp_max_c": 22.0, "precipitation_mm": 15,
         "plant_density_per_m2": 58, "soil_moisture_pct": 30, "leaf_defoliation_pct": 18,
         "waterlogged_days_consecutive": 0},
        {"day_of_season": 28, "soil_temp_c": 15.0, "air_temp_max_c": 24.0, "precipitation_mm": 5,
         "plant_density_per_m2": 58, "soil_moisture_pct": 22, "leaf_defoliation_pct": 27,
         "waterlogged_days_consecutive": 0},
        {"day_of_season": 55, "soil_temp_c": 18.0, "air_temp_max_c": 30.5, "precipitation_mm": 3,
         "plant_density_per_m2": 58, "soil_moisture_pct": 25, "waterlogged_days_consecutive": 0},
        {"day_of_season": 65, "soil_temp_c": 17.0, "air_temp_max_c": 25.0, "precipitation_mm": 18,
         "plant_density_per_m2": 58, "soil_moisture_pct": 38, "waterlogged_days_consecutive": 3},
        {"day_of_season": 110, "soil_temp_c": 14.0, "air_temp_max_c": 21.0, "precipitation_mm": 5,
         "plant_density_per_m2": 58, "soil_moisture_pct": 20, "seed_moisture_pct": 32,
         "waterlogged_days_consecutive": 0},
    ]

    summary = engine.run_season(state, sensor_log, weather=weather, latitude=lat)

    print(f"\nField: {summary['field_id']}  |  {summary['species']} ({summary['cultivar_type']})")
    print(f"Seeding: {summary['seeding_date']}  ->  Est. harvest: {summary['estimated_harvest_date']}")
    print(f"Preceding crop: {summary['preceding_crop']}")
    print(f"\nYield (calibrated process model x management):")
    print(f"  {summary['yield_potential_t_ha']} t/ha  ({summary['yield_potential_bu_ac']} bu/ac)")
    print(f"  breakdown: {summary['yield_breakdown']}")
    print(f"\nAlerts — CRITICAL: {summary['critical_count']} | "
          f"WARNING: {summary['warning_count']} | INFO: {summary['info_count']}")
    for a in summary.get("critical_alerts", []):
        print(f"  [!] {a}")
    for a in summary.get("warning_alerts", []):
        print(f"  [~] {a}")

    print("\n-- Seeding rate calculator --")
    print(json.dumps(calculate_seeding_rate(65, 5.5), indent=2))
    print("\n-- N requirement estimate --")
    print(json.dumps(estimate_n_requirement(3.5, CultivarType.HYBRID), indent=2))
    print("\n-- Harvest strategy --")
    print(json.dumps(get_harvest_strategy(state), indent=2))


if __name__ == "__main__":
    main()
