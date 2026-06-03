"""Wheat advisory demo: Zadoks-stage alerts + calibrated wheat-model yield & protein.

    python scripts/run_wheat_advisory.py
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from canola_dt.advisory import (
    WheatAdvisoryEngine,
    WheatClass,
    WheatFieldState,
    WheatPrecedingCrop,
    wheat_n_requirement,
    wheat_seeding_rate,
)
from canola_dt.config import load_config
from canola_dt.data import eccc


def _season_weather(cfg, year=2020):
    stations = eccc.station_map(cfg)
    sid, info = next((s, i) for s, i in stations.items() if i["province"] == "Saskatchewan")
    daily = eccc.fetch_daily(int(sid), year, cfg.path("data_raw") / "eccc")
    s = daily[(daily["date"] >= pd.Timestamp(year, 5, 1))
              & (daily["date"] <= pd.Timestamp(year, 10, 31))].set_index("date").asfreq("D")
    s[["tmin_c", "tmax_c", "tmean_c"]] = s[["tmin_c", "tmax_c", "tmean_c"]].interpolate(limit=5)
    s["precip_mm"] = s["precip_mm"].fillna(0.0)
    return s.reset_index(), float(info["lat"])


def main() -> None:
    print("=" * 66)
    print("  Wheat Digital Twin — advisory layer + calibrated yield & protein")
    print("=" * 66)
    cfg = load_config()
    engine = WheatAdvisoryEngine.with_calibrated_model(cfg)
    weather, lat = _season_weather(cfg, 2020)

    state = WheatFieldState(
        field_id="SK-WHEAT-2020", wheat_class=WheatClass.CWRS,
        seeding_date=date(2020, 5, 5), preceding_crop=WheatPrecedingCrop.WHEAT,  # wheat-on-wheat!
        years_since_last_wheat=1, latitude=lat,
        target_population_per_m2=275, n_applied_kg_per_ha=95, p2o5_applied_kg_per_ha=40,
        s_applied_kg_per_ha=12,
    )

    sensor_log = [
        {"day_of_season": 5, "air_temp_max_c": 16, "precipitation_mm": 30, "plant_population_per_m2": 0,
         "soil_moisture_pct": 28, "month": 5},
        {"day_of_season": 20, "air_temp_max_c": 20, "precipitation_mm": 15, "plant_population_per_m2": 210,
         "soil_moisture_pct": 26, "month": 5},
        {"day_of_season": 45, "air_temp_max_c": 24, "precipitation_mm": 12, "plant_population_per_m2": 210,
         "soil_moisture_pct": 30, "leaf_disease_severity_pct": 8, "month": 6},
        {"day_of_season": 53, "air_temp_max_c": 26, "precipitation_mm": 8, "plant_population_per_m2": 210,
         "soil_moisture_pct": 28, "midge_per_head": 0.3, "month": 6},
        {"day_of_season": 66, "air_temp_max_c": 23, "precipitation_mm": 18, "plant_population_per_m2": 210,
         "soil_moisture_pct": 38, "relative_humidity_pct": 78, "month": 7},  # FHB-favourable anthesis
        {"day_of_season": 105, "air_temp_max_c": 22, "precipitation_mm": 4, "plant_population_per_m2": 210,
         "soil_moisture_pct": 20, "grain_moisture_pct": 16, "lodging_pct": 15, "month": 8},
    ]

    summary = engine.run_season(state, sensor_log, weather=weather, latitude=lat)

    print(f"\nField: {summary['field_id']}  |  {summary['wheat_class']}  | "
          f"preceding: {summary['preceding_crop']}")
    print(f"Seeding {summary['seeding_date']} -> est. harvest {summary['estimated_harvest_date']}")
    print(f"\nYield: {summary['yield_potential_t_ha']} t/ha ({summary['yield_potential_bu_ac']} bu/ac)"
          f"  |  est. protein {summary['estimated_protein_pct']}%")
    print(f"  breakdown: {summary['yield_breakdown']}")
    print(f"\nAlerts — CRITICAL {summary['critical_count']} | WARNING {summary['warning_count']} | "
          f"INFO {summary['info_count']}")
    for a in summary["critical_alerts"]:
        print(f"  [!] {a}")
    for a in summary["warning_alerts"]:
        print(f"  [~] {a}")

    print("\n-- Wheat seeding-rate calculator (CWRS) --")
    print(json.dumps(wheat_seeding_rate(target_plants_per_m2=275, kernel_weight_g=37), indent=2))
    print("\n-- Wheat N requirement (4 t/ha, 13.5% protein) --")
    print(json.dumps(wheat_n_requirement(4.0, 13.5, soil_n_kg_per_ha=40), indent=2))

    print("\n-- Fertility report (N/P/K/S for a 4 t/ha target) --")
    print(json.dumps(engine.fertility_report(state, target_yield_t_ha=4.0), indent=2))


if __name__ == "__main__":
    main()
