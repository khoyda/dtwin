"""Calibrate the process-model parameters against trend-adjusted StatCan yields.

    python scripts/calibrate_process_model.py

Reports default-vs-calibrated fit, writes the best parameters to
artifacts/calibrated_params.json (which then overrides config defaults wherever
CanolaParameters.from_calibrated is used).
"""

from __future__ import annotations

import json
import time

import pandas as pd

from canola_dt import calibration as cal
from canola_dt.config import load_config
from canola_dt.simulation.process_model import CanolaParameters


def _fmt(m: dict) -> str:
    return (f"anomaly_RMSE={m['anomaly_rmse']:.0f}  anomaly_corr={m['anomaly_corr']:+.2f}  "
            f"| raw RMSE={m['rmse']:.0f} bias={m['bias']:+.0f}  (n={m['n']})")


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()
    base = CanolaParameters.from_config(cfg)

    print("== Loading weather + yield data ==")
    frames = cal.load_season_frames(cfg)
    targets = cal.load_targets(cfg)
    print(f"  province-years simulated : {len(frames)}")
    slopes = targets.groupby("province")["slope"].first()
    print("  technology trend (kg/ha/yr):", {p: round(v, 1) for p, v in slopes.items()})

    print("\n== Default parameters ==")
    base_metrics, _ = cal.evaluate(frames, targets, base)
    print("  " + _fmt(base_metrics))

    print("\n== Three-step calibration ==")
    t0 = time.time()
    out = cal.calibrate(cfg, base)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\n  step 1 — pattern (kl x heat), top 5 by anomaly correlation:")
    print(out["pattern_table"].head(5).to_string(index=False))
    d = out["diagnostics"]
    print(f"\n  step 2 — level: rue set to {out['params']['rue']} to match mean yield")
    print(f"           volatility diagnostic: sim anomaly std {d['sim_anom_std']:.0f} vs "
          f"provincial {d['obs_anom_std']:.0f} kg/ha (ratio {d['volatility_ratio']}x -- a "
          f"point sim is\n           intrinsically more variable than a provincial average; "
          f"variance is NOT calibrated)")
    print("\n  step 3 — per-province offset (adjusted obs - sim; representativeness):")
    for prov, v in out["offsets"].items():
        print(f"    {prov:<14} {v:+7.0f} kg/ha")

    print("\n== Calibrated parameters ==")
    print(f"  {out['params']}")
    print("  " + _fmt(out["metrics"]))
    c = out["corrected"]
    print(f"  absolute error after offsets: RMSE={c['rmse']:.0f}  MAE={c['mae']:.0f} kg/ha")
    print(f"  (default anomaly_corr {base_metrics['anomaly_corr']:+.2f} -> "
          f"calibrated {out['metrics']['anomaly_corr']:+.2f})")

    path = cal.save_calibrated(cfg, out["params"])
    report = path.parent / "calibration_report.json"
    report.write_text(json.dumps({
        "calibrated_params": {k: float(v) for k, v in out["params"].items()},
        "metrics": out["metrics"],
        "corrected_metrics": c,
        "diagnostics": d,
        "province_offsets_kg_ha": {p: round(float(v), 1) for p, v in out["offsets"].items()},
        "default_metrics": base_metrics,
    }, indent=2))
    print(f"\n  saved params  -> {path}")
    print(f"  saved report  -> {report}")
    print("  (params loaded automatically via CanolaParameters.from_calibrated)")


if __name__ == "__main__":
    main()
