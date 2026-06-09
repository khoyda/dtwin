"""Yellow-pea advisory demo: legume alerts + calibrated yield & protein.

    python scripts/run_pea_advisory.py
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from canola_dt.advisory import (
    PeaAdvisoryEngine,
    PeaFieldState,
    PeaPrecedingCrop,
    PeaType,
    pea_seeding_rate,
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
    print("  Yellow-Pea Digital Twin — advisory layer + calibrated yield")
    print("=" * 66)
    cfg = load_config()
    engine = PeaAdvisoryEngine.with_calibrated_model(cfg)
    weather, lat = _season_weather(cfg, 2020)

    state = PeaFieldState(
        field_id="SK-PEA-2020", pea_type=PeaType.YELLOW, seeding_date=date(2020, 5, 5),
        preceding_crop=PeaPrecedingCrop.PEA, years_since_last_pulse=1,  # short pulse rotation!
        latitude=lat, inoculant_applied=False,            # forgot inoculant!
        n_applied_kg_per_ha=60,                            # too much N — suppresses fixation
        p2o5_applied_kg_per_ha=40, s_applied_kg_per_ha=8,
    )

    sensor_log = [
        {"day_of_season": 4, "air_temp_max_c": 16, "precipitation_mm": 15,
         "plant_population_per_m2": 0, "soil_moisture_pct": 28},     # planning alerts
        {"day_of_season": 30, "air_temp_max_c": 20, "precipitation_mm": 12,
         "plant_population_per_m2": 78, "soil_moisture_pct": 30, "weevil_damage_pct": 35},
        {"day_of_season": 52, "air_temp_max_c": 28, "precipitation_mm": 8,
         "plant_population_per_m2": 78, "soil_moisture_pct": 25, "aphids_per_tip": 4},  # heat + aphids
        {"day_of_season": 98, "air_temp_max_c": 22, "precipitation_mm": 4,
         "plant_population_per_m2": 78, "soil_moisture_pct": 20, "pod_brown_pct": 35,
         "grain_moisture_pct": 18, "lodging_pct": 20},
    ]

    summary = engine.run_season(state, sensor_log, weather=weather, latitude=lat)
    print(f"\nField: {summary['field_id']}  |  {summary['pea_type']} pea  | "
          f"preceding: {summary['preceding_crop']}")
    print(f"\nYield: {summary['yield_potential_t_ha']} t/ha ({summary['yield_potential_bu_ac']} bu/ac)"
          f"  |  protein {summary['estimated_protein_pct']}%")
    print(f"  breakdown: {summary['yield_breakdown']}")
    print(f"\nAlerts — CRITICAL {summary['critical_count']} | WARNING {summary['warning_count']} | "
          f"INFO {summary['info_count']}")
    for a in summary["critical_alerts"]:
        print(f"  [!] {a}")
    for a in summary["warning_alerts"]:
        print(f"  [~] {a}")

    print("\n-- Pea seeding-rate calculator (large seed) --")
    print(json.dumps(pea_seeding_rate(target_plants_per_m2=80, thousand_kernel_weight_g=235), indent=2))
    print("\n-- Fertility report (3.0 t/ha target — note N is fixed) --")
    print(json.dumps(engine.fertility_report(state, 3.0), indent=2))


if __name__ == "__main__":
    main()
