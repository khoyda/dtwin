"""Barley advisory demo: Zadoks alerts + calibrated yield, protein & malt grade.

    python scripts/run_barley_advisory.py
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from canola_dt.advisory import (
    BarleyAdvisoryEngine,
    BarleyFieldState,
    BarleyPrecedingCrop,
    BarleyType,
    barley_seeding_rate,
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
    print("  Barley Digital Twin — advisory layer + calibrated yield & malt grade")
    print("=" * 66)
    cfg = load_config()
    engine = BarleyAdvisoryEngine.with_calibrated_model(cfg)
    weather, lat = _season_weather(cfg, 2020)

    state = BarleyFieldState(
        field_id="SK-MALT-2020", barley_type=BarleyType.MALT_2ROW,
        seeding_date=date(2020, 5, 5), preceding_crop=BarleyPrecedingCrop.BARLEY,  # barley-on-barley
        years_since_last_barley=1, latitude=lat,
        target_population_per_m2=250, n_applied_kg_per_ha=130,  # high N -> protein risk
        p2o5_applied_kg_per_ha=40, s_applied_kg_per_ha=12,
    )

    sensor_log = [
        {"day_of_season": 5, "air_temp_max_c": 16, "precipitation_mm": 20,
         "plant_population_per_m2": 0, "soil_moisture_pct": 28},   # planning: rotation + malt-N
        {"day_of_season": 18, "air_temp_max_c": 20, "precipitation_mm": 15,
         "plant_population_per_m2": 230, "soil_moisture_pct": 26},
        {"day_of_season": 33, "air_temp_max_c": 22, "precipitation_mm": 12,
         "plant_population_per_m2": 230, "soil_moisture_pct": 30, "scald_severity_pct": 3},
        {"day_of_season": 46, "air_temp_max_c": 24, "precipitation_mm": 10,
         "plant_population_per_m2": 230, "soil_moisture_pct": 28, "net_blotch_severity_pct": 8},
        {"day_of_season": 64, "air_temp_max_c": 23, "precipitation_mm": 18,
         "plant_population_per_m2": 230, "soil_moisture_pct": 38, "relative_humidity_pct": 78},
        {"day_of_season": 96, "air_temp_max_c": 22, "precipitation_mm": 4,
         "plant_population_per_m2": 230, "soil_moisture_pct": 20, "grain_moisture_pct": 16,
         "lodging_pct": 12},
    ]

    summary = engine.run_season(state, sensor_log, weather=weather, latitude=lat)

    print(f"\nField: {summary['field_id']}  |  {summary['barley_type']}  | "
          f"preceding: {summary['preceding_crop']}")
    print(f"\nYield: {summary['yield_potential_t_ha']} t/ha ({summary['yield_potential_bu_ac']} bu/ac)"
          f"  |  protein {summary['estimated_protein_pct']}%  |  "
          f"malt grade {'OK' if summary['malt_grade_ok'] else 'FAIL -> feed'}")
    print(f"  breakdown: {summary['yield_breakdown']}")
    print(f"\nAlerts — CRITICAL {summary['critical_count']} | WARNING {summary['warning_count']} | "
          f"INFO {summary['info_count']}")
    for a in summary["critical_alerts"]:
        print(f"  [!] {a}")
    for a in summary["warning_alerts"]:
        print(f"  [~] {a}")

    print("\n-- Barley seeding-rate calculator --")
    print(json.dumps(barley_seeding_rate(target_plants_per_m2=250, thousand_kernel_weight_g=45), indent=2))
    print("\n-- Fertility report (4.0 t/ha target) --")
    print(json.dumps(engine.fertility_report(state, 4.0), indent=2))


if __name__ == "__main__":
    main()
