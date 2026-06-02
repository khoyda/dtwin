"""Run the APSIM-style canola process model on real ECCC weather.

    python scripts/run_process_model.py [year]

Pulls a full growing season from the SK station (Indian Head CDA), runs the
mechanistic crop model, and prints the simulated phenology, canopy, water balance
and yield. Uses cached ECCC data if present.
"""

from __future__ import annotations

import sys

import pandas as pd

from canola_dt.config import load_config
from canola_dt.constants import GrowthStage
from canola_dt.data import eccc
from canola_dt.simulation.process_model import CanolaCropModel, CanolaParameters


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2020

    # First SK station + latitude.
    stations = eccc.station_map(cfg)
    station_id, info = next(
        (sid, i) for sid, i in stations.items() if i["province"] == "Saskatchewan"
    )
    cache = cfg.path("data_raw") / "eccc"

    # Sowing window: May 1 through Oct 31 (covers emergence -> maturity).
    daily = eccc.fetch_daily(int(station_id), year, cache)
    season = daily[(daily["date"] >= pd.Timestamp(year, 5, 1))
                   & (daily["date"] <= pd.Timestamp(year, 10, 31))].reset_index(drop=True)
    # Fill any short gaps so the daily loop is continuous.
    season = season.set_index("date").asfreq("D")
    season[["tmin_c", "tmax_c", "tmean_c"]] = season[["tmin_c", "tmax_c", "tmean_c"]].interpolate(limit=5)
    season["precip_mm"] = season["precip_mm"].fillna(0.0)
    season = season.reset_index()

    print(f"== APSIM-style canola model — {info['name']} ({info['province']}), {year} ==")
    print(f"   site lat {info['lat']}, season rows {len(season)} "
          f"({season['date'].min().date()} -> {season['date'].max().date()})")

    # Use calibrated parameters if a calibration run has produced them.
    model = CanolaCropModel(CanolaParameters.from_calibrated(cfg))
    result = model.run(season, latitude_deg=float(info["lat"]))

    s = result.summary
    print("\n-- season summary --")
    for k, v in s.items():
        print(f"   {k:<28} {v}")

    print("\n-- phenology (days from sowing) --")
    timeline = (
        result.daily.groupby("stage")["date"].first().reset_index().sort_values("stage")
    )
    start = result.daily["date"].iloc[0]
    for _, r in timeline.iterrows():
        d = (r["date"] - start).days
        print(f"   {GrowthStage(int(r['stage'])).name:<10} {r['date'].date()} (day {d})")

    print("\n-- canopy / water at 10-day intervals --")
    cols = ["date", "stage", "tt_cum", "lai", "biomass_g_m2", "profile_paw_mm", "water_stress"]
    sample = result.daily.iloc[::10].copy()
    sample["stage"] = sample["stage"].map(lambda x: GrowthStage(int(x)).name)
    print(sample[cols].to_string(index=False))


if __name__ == "__main__":
    main()
