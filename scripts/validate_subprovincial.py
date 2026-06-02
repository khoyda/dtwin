"""Validate the calibrated process model against SK RM-level (SCIC) canola yields.

    python scripts/validate_subprovincial.py

Tests whether matching each station to its local Rural Municipality yield beats the
provincial-scale skill ceiling (anomaly correlation ~0.39).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canola_dt import subprovincial as sp
from canola_dt.config import load_config


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()

    print("== Sub-provincial validation: SK stations vs nearest-RM canola yields ==")
    pairs = sp.build_station_rm_pairs(cfg)
    print(f"  station-year pairs : {len(pairs)} across {pairs['station_id'].nunique()} SK stations")
    print(f"  RM yield coverage  : {pairs['year'].min()}-{pairs['year'].max()}")

    res = sp.local_vs_provincial(cfg)
    print("\n  station -> nearest RM:")
    for sid, rmno in res["station_rm"].items():
        print(f"    station {sid} -> RM {rmno}")

    lc = res["local_anomaly_corr"]
    pc = res["provincial_anomaly_corr"]
    print("\n== Interannual anomaly correlation (same stations, same detrending) ==")
    print(f"  vs PROVINCIAL SK yield : {pc:+.3f}")
    print(f"  vs LOCAL RM yield      : {lc:+.3f}")
    delta = lc - pc
    verdict = "local matching improves skill" if delta > 0 else "no local improvement"
    print(f"  delta                  : {delta:+.3f}  ({verdict})")

    # Per-station local correlation, for transparency.
    print("\n  per-station local correlation (sim anomaly vs RM-yield anomaly):")
    for sid, g in pairs.groupby("station_id"):
        if len(g) < 5:
            continue
        slope, intercept = np.polyfit(g["year"], g["rm_yield"], 1)
        oa = g["rm_yield"] - (intercept + slope * g["year"])
        sa = g["sim_yield"] - g["sim_yield"].mean()
        r = np.corrcoef(sa, oa)[0, 1]
        print(f"    station {sid} (RM {g['rmno'].iloc[0]}, n={len(g)}): {r:+.2f}")


if __name__ == "__main__":
    main()
