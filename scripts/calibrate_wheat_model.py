"""Calibrate the spring-wheat process model against trend-adjusted StatCan yields.

    python scripts/calibrate_wheat_model.py

Writes artifacts/wheat_calibrated_params.json (loaded via WheatParameters.from_calibrated).
"""

from __future__ import annotations

import json
import time

import pandas as pd

from canola_dt import wheat_calibration as wc
from canola_dt.config import load_config
from canola_dt.simulation.wheat_model import WheatParameters


def _fmt(m: dict) -> str:
    return (f"anomaly_RMSE={m['anomaly_rmse']:.0f}  anomaly_corr={m['anomaly_corr']:+.2f}  "
            f"| raw RMSE={m['rmse']:.0f} bias={m['bias']:+.0f}  (n={m['n']})")


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()
    base = WheatParameters.from_config(cfg)

    print("== Spring-wheat calibration vs StatCan 'Wheat, spring' yields ==")
    targets = wc.load_targets(cfg)
    slopes = targets.groupby("province")["slope"].first()
    print("  technology trend (kg/ha/yr):", {p: round(v, 1) for p, v in slopes.items()})

    base_metrics, _ = wc.evaluate(wc.cal.load_season_frames(cfg), targets, base)
    print("  default params: " + _fmt(base_metrics))

    print("\n== Three-step calibration ==")
    t0 = time.time()
    out = wc.calibrate(cfg, base)
    print(f"  done in {time.time() - t0:.1f}s  (province-years: {out['n']})")
    print("\n  step 1 — pattern (kl x heat), top 5 by anomaly correlation:")
    print(out["pattern_table"].head(5).to_string(index=False))
    d = out["diagnostics"]
    print(f"\n  step 2 — level: rue = {out['params']['rue']}  "
          f"(volatility ratio {d['volatility_ratio']}x)")
    print("  step 3 — per-province offset (adjusted obs - sim):")
    for prov, v in out["offsets"].items():
        print(f"    {prov:<14} {v:+7.0f} kg/ha")

    print("\n== Calibrated parameters ==")
    print(f"  {out['params']}")
    print("  " + _fmt(out["metrics"]))
    c = out["corrected"]
    print(f"  absolute error after offsets: RMSE={c['rmse']:.0f}  MAE={c['mae']:.0f} kg/ha")
    print(f"  (default anomaly_corr {base_metrics['anomaly_corr']:+.2f} -> "
          f"calibrated {out['metrics']['anomaly_corr']:+.2f})")

    path = wc.save_calibrated(cfg, out["params"])
    report = path.parent / "wheat_calibration_report.json"
    report.write_text(json.dumps({
        "calibrated_params": {k: float(v) for k, v in out["params"].items()},
        "metrics": out["metrics"], "corrected_metrics": c, "diagnostics": d,
        "province_offsets_kg_ha": {p: round(float(v), 1) for p, v in out["offsets"].items()},
        "default_metrics": base_metrics,
    }, indent=2))
    print(f"\n  saved params -> {path}")
    print(f"  saved report -> {report}")


if __name__ == "__main__":
    main()
