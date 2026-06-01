"""Calibrate the APSIM-style process model against trend-adjusted StatCan yields.

The process model has fixed genetics, so it cannot reproduce the multi-decade
technology trend in observed yields. We therefore:

1. Fit a per-province linear **technology trend** to observed yield vs. year and
   express every year's observed yield at a common reference year ("what this
   year's *weather* would have yielded with reference-year genetics").
2. Calibrate a small, identifiable parameter set (RUE = yield level, ``kl`` =
   drought sensitivity, ``hi_heat_sensitivity`` = heat penalty) by grid search to
   minimize RMSE between simulated and trend-adjusted observed yields.

This separates the exogenous technology trend (handled elsewhere, e.g. the ML
model's ``year`` feature) from the model's biophysical weather response, which is
what a process model should capture.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from canola_dt.config import Config, load_config
from canola_dt.data import eccc, statcan
from canola_dt.simulation.process_model import CALIBRATABLE, CanolaCropModel, CanolaParameters

SEASON_START = (5, 1)
SEASON_END = (10, 31)
MIN_SEASON_DAYS = 120
MIN_COMPLETENESS = 0.80


# --- pure helpers (unit-tested) ---------------------------------------------

def technology_detrend(observed: pd.DataFrame, ref_year: int) -> pd.DataFrame:
    """Add per-province linear trend and yields adjusted to ``ref_year`` genetics.

    ``observed`` needs columns ``province, year, yield_kg_ha``. Returns it with
    added ``trend`` (fitted), ``slope`` (kg/ha/yr) and ``adjusted`` columns, where
    ``adjusted = yield + slope * (ref_year - year)``.
    """
    out = []
    for _, g in observed.groupby("province"):
        g = g.sort_values("year").copy()
        slope, intercept = np.polyfit(g["year"], g["yield_kg_ha"], 1)
        g["slope"] = slope
        g["trend"] = intercept + slope * g["year"]
        g["adjusted"] = g["yield_kg_ha"] + slope * (ref_year - g["year"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def calibration_metrics(merged: pd.DataFrame) -> dict:
    """Interannual (province-centered) weather-skill metrics plus raw level error.

    The calibration objective is ``anomaly_rmse`` — the error in year-to-year
    deviations *after removing each province's mean* — so the point-vs-province
    level bias (a representativeness artifact) cannot hijack the biophysical
    parameters. Raw ``rmse``/``bias`` are reported for reference only.

    ``merged`` needs ``province, sim_yield, adjusted, yield_kg_ha, trend``.
    """
    err = merged["sim_yield"] - merged["adjusted"]
    obs_anom = merged["yield_kg_ha"] - merged["trend"]   # detrended weather residual
    sim_anom = merged["sim_yield"] - merged.groupby("province")["sim_yield"].transform("mean")
    anom_err = sim_anom - obs_anom
    corr = np.corrcoef(sim_anom, obs_anom)[0, 1] if len(merged) > 2 else float("nan")
    return {
        "anomaly_rmse": float(np.sqrt((anom_err**2).mean())),
        "anomaly_corr": float(corr),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "n": int(len(merged)),
    }


def province_offsets(merged: pd.DataFrame) -> pd.Series:
    """Per-province representativeness offset = mean(adjusted obs - sim yield)."""
    return (merged["adjusted"] - merged["sim_yield"]).groupby(merged["province"]).mean()


def corrected_metrics(merged: pd.DataFrame, offsets: pd.Series) -> dict:
    """Absolute error after applying per-province offsets (sim + offset vs obs)."""
    pred = merged["sim_yield"] + merged["province"].map(offsets)
    err = pred - merged["adjusted"]
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
    }


# --- data assembly -----------------------------------------------------------

def _prepare_season(daily: pd.DataFrame, year: int) -> pd.DataFrame:
    s = daily[(daily["date"] >= pd.Timestamp(year, *SEASON_START))
              & (daily["date"] <= pd.Timestamp(year, *SEASON_END))]
    s = s.set_index("date").asfreq("D")
    s[["tmin_c", "tmax_c", "tmean_c"]] = s[["tmin_c", "tmax_c", "tmean_c"]].interpolate(limit=5)
    s["precip_mm"] = s["precip_mm"].fillna(0.0)
    return s.reset_index()


def load_season_frames(cfg: Config) -> dict[tuple[str, int], tuple[pd.DataFrame, float]]:
    """Pre-load usable (province, year) -> (season frame, latitude). Cached on disk."""
    ds = cfg["data_sources"]
    cache = cfg.path("data_raw") / "eccc"
    frames: dict[tuple[str, int], tuple[pd.DataFrame, float]] = {}
    for sid, info in ds["eccc"]["stations"].items():
        for year in range(ds["start_year"], ds["end_year"] + 1):
            frame = _prepare_season(eccc.fetch_daily(int(sid), year, cache), year)
            if len(frame) < MIN_SEASON_DAYS or frame["tmean_c"].notna().mean() < MIN_COMPLETENESS:
                continue
            frames[(info["province"], year)] = (frame, float(info["lat"]))
    return frames


def simulate_all(frames: dict, params: CanolaParameters) -> pd.DataFrame:
    """Run the process model over every pre-loaded (province, year)."""
    rows = []
    for (province, year), (frame, lat) in frames.items():
        res = CanolaCropModel(params).run(frame, lat)
        rows.append({"province": province, "year": year,
                     "sim_yield": res.summary["yield_kg_ha"]})
    return pd.DataFrame(rows)


def load_targets(cfg: Config) -> pd.DataFrame:
    """StatCan canola yields within the window, with technology-detrend columns."""
    ds = cfg["data_sources"]
    obs = statcan.load_canola_yield(
        pid=ds["statcan"]["table_pid"],
        cache_dir=cfg.path("data_raw") / "statcan",
        provinces=ds["statcan"]["provinces"],
        crop=ds["statcan"]["crop"],
        yield_disposition=ds["statcan"]["yield_disposition"],
    )
    obs = obs[(obs["year"] >= ds["start_year"]) & (obs["year"] <= ds["end_year"])]
    return technology_detrend(obs, ref_year=ds["end_year"])


def evaluate(frames: dict, targets: pd.DataFrame, params: CanolaParameters):
    """Simulate, merge with targets, return (metrics, merged frame)."""
    sim = simulate_all(frames, params)
    merged = sim.merge(
        targets[["province", "year", "adjusted", "yield_kg_ha", "trend"]],
        on=["province", "year"],
    )
    return calibration_metrics(merged), merged


# --- grid-search calibration -------------------------------------------------

# Pattern (sensitivity) grid for step 1. RUE is set analytically (step 2), not gridded.
DEFAULT_GRID = {
    "kl": [0.04, 0.06, 0.08, 0.10, 0.14],            # drought sensitivity (pattern)
    "hi_heat_sensitivity": [0.0, 0.03, 0.06, 0.09],  # heat penalty (pattern)
}


def _anom_std(merged: pd.DataFrame) -> tuple[float, float]:
    """(sim, observed) interannual anomaly standard deviations (province-centered)."""
    sim = (merged["sim_yield"] - merged.groupby("province")["sim_yield"].transform("mean")).std()
    obs = (merged["yield_kg_ha"] - merged["trend"]).std()
    return float(sim), float(obs)


def calibrate(cfg: Config, base: CanolaParameters, grid: dict | None = None) -> dict:
    """Three-step calibration, each step identifying only what it can.

    1. **Pattern** — grid ``kl`` x ``hi_heat_sensitivity`` to maximize the
       interannual anomaly correlation (which good/bad years; insensitive to level
       and scale, so it isolates the biophysical sensitivities).
    2. **Level** — set ``rue`` so the overall mean simulated yield matches the mean
       trend-adjusted observed yield. (Note: we deliberately do *not* match the
       *variance* — a point-scale simulation is intrinsically more volatile than a
       province-wide yield average; see the reported ``volatility_ratio`` diagnostic.)
    3. **Per-province offsets** — residual level differences (point station != province).

    Returns a dict with calibrated params, metrics, offsets and the pattern table.
    """
    grid = grid or DEFAULT_GRID
    frames = load_season_frames(cfg)
    targets = load_targets(cfg)

    # --- step 1: pattern parameters ---
    rows, best = [], None
    for kl, hs in itertools.product(grid["kl"], grid["hi_heat_sensitivity"]):
        ov = {"kl": kl, "hi_heat_sensitivity": hs}
        m, _ = evaluate(frames, targets, base.with_overrides(ov))
        rows.append({**ov, "anomaly_corr": m["anomaly_corr"], "anomaly_rmse": m["anomaly_rmse"]})
        corr = -1.0 if m["anomaly_corr"] != m["anomaly_corr"] else m["anomaly_corr"]
        if best is None or corr > best[1]:
            best = (ov, corr)
    pattern = best[0]

    # --- step 2: level (rue) to match overall mean adjusted-observed yield ---
    _, merged0 = evaluate(frames, targets, base.with_overrides(pattern))
    sim_std, obs_std = _anom_std(merged0)
    rue_star = float(np.clip(
        base.rue * merged0["adjusted"].mean() / max(merged0["sim_yield"].mean(), 1e-6),
        1.0, 3.5,
    ))
    params = {"rue": round(rue_star, 2), **pattern}

    # --- step 3: per-province offsets + final metrics ---
    metrics, merged = evaluate(frames, targets, base.with_overrides(params))
    offsets = province_offsets(merged)
    pattern_table = (
        pd.DataFrame(rows).sort_values("anomaly_corr", ascending=False).reset_index(drop=True)
    )
    return {
        "params": params,
        "metrics": metrics,
        "offsets": offsets,
        "corrected": corrected_metrics(merged, offsets),
        "diagnostics": {
            "sim_anom_std": round(sim_std, 1),
            "obs_anom_std": round(obs_std, 1),
            # >1 means the point model is more volatile than the provincial average.
            "volatility_ratio": round(sim_std / max(obs_std, 1e-6), 2),
        },
        "pattern_table": pattern_table,
        "n": len(frames),
    }


def save_calibrated(cfg: Config, overrides: dict) -> Path:
    """Persist calibrated parameters; picked up by CanolaParameters.from_calibrated."""
    path = cfg.path("artifacts") / "calibrated_params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: float(v) for k, v in overrides.items() if k in CALIBRATABLE}
    path.write_text(json.dumps(payload, indent=2))
    return path
