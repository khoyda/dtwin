"""Validate the calibrated spring-barley model vs SK RM-level (SCIC) barley yields.

    python scripts/validate_barley_subprovincial.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canola_dt import barley_subprovincial as bsp
from canola_dt.config import load_config


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()

    print("== Barley sub-provincial validation: SK stations vs nearest-RM barley yields ==")
    res = bsp.local_vs_provincial(cfg)
    pairs = res["pairs"]
    print(f"  station-year pairs : {len(pairs)} across {res['n_stations']} SK stations "
          f"({pairs['year'].min()}-{pairs['year'].max()}; 2021 excluded from fitting)")

    lc, pc = res["local_anomaly_corr"], res["provincial_anomaly_corr"]
    print("\n== Interannual anomaly correlation (same stations, same detrending) ==")
    print(f"  vs PROVINCIAL SK barley yield : {pc:+.3f}")
    print(f"  vs LOCAL RM barley yield      : {lc:+.3f}")
    print(f"  delta                         : {lc - pc:+.3f}  "
          f"({'local matching improves skill' if lc > pc else 'no local improvement'})")

    print("\n  per-station local correlation (sim anomaly vs RM-yield anomaly):")
    for sid, g in pairs.groupby("station_id"):
        if len(g) < 5:
            continue
        slope, intercept = np.polyfit(g["year"], g["rm_yield"], 1)
        oa = g["rm_yield"] - (intercept + slope * g["year"])
        sa = g["sim_yield"] - g["sim_yield"].mean()
        print(f"    station {sid} (RM {g['rmno'].iloc[0]}, n={len(g)}): "
              f"{np.corrcoef(sa, oa)[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
