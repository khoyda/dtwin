"""Run the spring-barley process model on real ECCC weather for a season.

    python scripts/run_barley_model.py [year]
"""

from __future__ import annotations

import sys

import pandas as pd

from canola_dt.config import load_config
from canola_dt.data import eccc
from canola_dt.simulation.barley_model import BarleyCropModel, BarleyParameters, BarleyStage


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2020

    stations = eccc.station_map(cfg)
    sid, info = next((s, i) for s, i in stations.items() if i["province"] == "Saskatchewan")
    daily = eccc.fetch_daily(int(sid), year, cfg.path("data_raw") / "eccc")
    s = daily[(daily["date"] >= pd.Timestamp(year, 5, 1))
              & (daily["date"] <= pd.Timestamp(year, 10, 31))].set_index("date").asfreq("D")
    s[["tmin_c", "tmax_c", "tmean_c"]] = s[["tmin_c", "tmax_c", "tmean_c"]].interpolate(limit=5)
    s["precip_mm"] = s["precip_mm"].fillna(0.0)
    s = s.reset_index()

    print(f"== Spring-barley model — {info['name']} (SK), {year} ==")
    result = BarleyCropModel(BarleyParameters.from_calibrated(cfg)).run(s, float(info["lat"]))
    for k, v in result.summary.items():
        print(f"   {k:<30} {v}")

    print("\n-- phenology (days from sowing) --")
    start = result.daily["date"].iloc[0]
    for stage, grp in result.daily.groupby("stage"):
        d = (grp["date"].iloc[0] - start).days
        print(f"   {BarleyStage(int(stage)).name:<11} {grp['date'].iloc[0].date()} (day {d})")


if __name__ == "__main__":
    main()
