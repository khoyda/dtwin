"""Discover & screen ECCC stations per province; write a generated stations file.

Selects long-record, completeness-verified, spatially-spread daily-weather stations
in each province's canola belt and writes them to
``config/stations.generated.yaml`` (which the pipelines load in preference to the
inline list in config.yaml). Re-run with a larger ``--per-province`` to add stations.

    python scripts/discover_stations.py --per-province 10

Selection: filter inventory by province, daily-record coverage of the configured
window, canola-belt latitude (49-57 N) and non-mountain elevation; screen real
completeness on sample years (incl. a drought year); keep the best; then greedily
pick a spatially-spread subset (farthest-point sampling).
"""

from __future__ import annotations

import argparse
import math

import yaml

from canola_dt.config import load_config
from canola_dt.data import eccc

# Complete but non-representative of canola land (mountains, parks, boreal).
NAME_DENYLIST = ("KANANASKIS", "WASAGAMING", "FLIN FLON", "BANFF", "JASPER", "NORDEGG")
SCREEN_YEARS = (2000, 2008, 2015, 2021)   # 2021 = severe Prairie drought
POOL_PER_PROVINCE = 30                      # candidates screened before spatial selection
MIN_MEAN_COMPLETENESS = 0.95
MIN_MIN_COMPLETENESS = 0.85


def _distance(a: dict, b: dict) -> float:
    """Rough planar distance between two lat/lon points (degrees, lon scaled)."""
    scale = math.cos(math.radians((a["lat"] + b["lat"]) / 2))
    return math.hypot(a["lat"] - b["lat"], (a["lon"] - b["lon"]) * scale)


def _farthest_point_select(candidates: list[dict], n: int) -> list[dict]:
    """Greedily pick ``n`` stations maximizing spatial spread (most complete first)."""
    if not candidates:
        return []
    chosen = [candidates[0]]  # candidates arrive sorted by completeness desc
    while len(chosen) < n and len(chosen) < len(candidates):
        best, best_d = None, -1.0
        for c in candidates:
            if c in chosen:
                continue
            d = min(_distance(c, s) for s in chosen)
            if d > best_d:
                best, best_d = c, d
        chosen.append(best)
    return chosen


def discover(cfg, provinces, per_province: int) -> dict[int, dict]:
    ds = cfg["data_sources"]
    cache = cfg.path("data_raw") / "eccc"
    inv = eccc.load_station_inventory(cache)
    inv = inv.dropna(subset=["DLY First Year", "DLY Last Year"])

    selected: dict[int, dict] = {}
    for prov in provinces:
        cand = inv[
            (inv["Province"] == prov.upper())
            & (inv["DLY First Year"].astype(int) <= ds["start_year"])
            & (inv["DLY Last Year"].astype(int) >= ds["end_year"])
            & (inv["Latitude (Decimal Degrees)"].between(49, 57))
            & (inv["Elevation (m)"].fillna(0) <= 1100)
            & (~inv["Name"].str.contains("|".join(NAME_DENYLIST), case=False, na=False))
        ].copy()
        cand["span"] = cand["DLY Last Year"].astype(int) - cand["DLY First Year"].astype(int)
        cand = cand.sort_values("span", ascending=False).head(POOL_PER_PROVINCE)

        screened = []
        for _, r in cand.iterrows():
            mn, mean = eccc.season_completeness_for(r["Station ID"], SCREEN_YEARS, cache)
            if mean >= MIN_MEAN_COMPLETENESS and mn >= MIN_MIN_COMPLETENESS:
                screened.append({
                    "id": int(r["Station ID"]),
                    "province": prov,
                    "name": str(r["Name"]).strip()[:24],
                    "lat": round(float(r["Latitude (Decimal Degrees)"]), 2),
                    "lon": round(float(r["Longitude (Decimal Degrees)"]), 2),
                    "mean_compl": round(mean, 3),
                })
        screened.sort(key=lambda s: s["mean_compl"], reverse=True)
        picks = _farthest_point_select(screened, per_province)
        print(f"  {prov:<14} {len(cand)} candidates -> {len(screened)} pass -> {len(picks)} selected")
        for p in picks:
            selected[p["id"]] = {"province": p["province"], "name": p["name"],
                                 "lat": p["lat"], "lon": p["lon"]}
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-province", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    provinces = cfg["data_sources"]["statcan"]["provinces"]
    print(f"== Discovering up to {args.per_province} stations/province "
          f"(screening years {SCREEN_YEARS}) ==")
    stations = discover(cfg, provinces, args.per_province)

    out = cfg.root / "config" / eccc.GENERATED_STATIONS
    # Plain dict dump keyed by station id; pipelines load via eccc.station_map().
    out.write_text(yaml.safe_dump(stations, sort_keys=True, default_flow_style=False),
                   encoding="utf-8")
    print(f"\n  total {len(stations)} stations -> {out}")
    print("  (loaded in preference to config.yaml via eccc.station_map)")


if __name__ == "__main__":
    main()
