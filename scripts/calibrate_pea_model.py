"""Calibrate the yellow-pea process model against trend-adjusted StatCan yields.

    python scripts/calibrate_pea_model.py
"""

from __future__ import annotations

import json
import time

import pandas as pd

from canola_dt import pea_calibration as pc
from canola_dt.config import load_config
from canola_dt.simulation.pea_model import PeaParameters


def _fmt(m: dict) -> str:
    return (f"anomaly_RMSE={m['anomaly_rmse']:.0f}  anomaly_corr={m['anomaly_corr']:+.2f}  "
            f"| raw RMSE={m['rmse']:.0f} bias={m['bias']:+.0f}  (n={m['n']})")


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()
    base = PeaParameters.from_config(cfg)

    print("== Yellow-pea calibration vs StatCan 'Peas, dry' yields ==")
    targets = pc.load_targets(cfg)
    slopes = targets.groupby("province")["slope"].first()
    print("  technology trend (kg/ha/yr):", {p: round(v, 1) for p, v in slopes.items()})
    base_metrics, _ = pc.evaluate(pc.cal.load_season_frames(cfg), targets, base)
    print("  default params: " + _fmt(base_metrics))

    print("\n== Three-step calibration ==")
    t0 = time.time()
    out = pc.calibrate(cfg, base)
    print(f"  done in {time.time() - t0:.1f}s  (province-years: {out['n']})")
    d = out["diagnostics"]
    print(f"  rue={out['params']['rue']}  kl={out['params']['kl']}  "
          f"heat={out['params']['hi_heat_sensitivity']}  (volatility {d['volatility_ratio']}x)")
    print("  per-province offset:", {p: round(float(v)) for p, v in out["offsets"].items()})
    print("  " + _fmt(out["metrics"]))
    c = out["corrected"]
    print(f"  absolute error after offsets: RMSE={c['rmse']:.0f}  MAE={c['mae']:.0f} kg/ha")
    print(f"  (default anomaly_corr {base_metrics['anomaly_corr']:+.2f} -> "
          f"calibrated {out['metrics']['anomaly_corr']:+.2f})")

    path = pc.save_calibrated(cfg, out["params"])
    (path.parent / "pea_calibration_report.json").write_text(json.dumps({
        "calibrated_params": {k: float(v) for k, v in out["params"].items()},
        "metrics": out["metrics"], "corrected_metrics": c, "diagnostics": d,
        "province_offsets_kg_ha": {p: round(float(v), 1) for p, v in out["offsets"].items()},
        "default_metrics": base_metrics,
    }, indent=2))
    print(f"\n  saved params -> {path}")


if __name__ == "__main__":
    main()
