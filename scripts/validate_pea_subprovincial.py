"""Validate the calibrated yellow-pea model vs SK RM-level (SCIC) pea yields.

    python scripts/validate_pea_subprovincial.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canola_dt import pea_subprovincial as psp
from canola_dt.config import load_config


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()
    print("== Pea sub-provincial validation: SK stations vs nearest-RM pea yields ==")
    res = psp.local_vs_provincial(cfg)
    pairs = res["pairs"]
    print(f"  station-year pairs : {len(pairs)} across {res['n_stations']} SK stations "
          f"({pairs['year'].min()}-{pairs['year'].max()}; 2021 excluded from fitting)")
    lc, pc = res["local_anomaly_corr"], res["provincial_anomaly_corr"]
    print("\n== Interannual anomaly correlation (same stations, same detrending) ==")
    print(f"  vs PROVINCIAL SK pea yield : {pc:+.3f}")
    print(f"  vs LOCAL RM pea yield      : {lc:+.3f}")
    print(f"  delta                      : {lc - pc:+.3f}  "
          f"({'local matching improves skill' if lc > pc else 'no local improvement'})")


if __name__ == "__main__":
    main()
