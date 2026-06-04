"""Scenario forecasting CLI for canola and wheat — test what-ifs for the season.

Single scenario (flags)::

    python scripts/forecast.py --crop wheat --n 110 --preceding canola --weather inseason
    python scripts/forecast.py --crop canola --n 150 --s 8 --weather analog --analog-year 2021

Batch comparison from a file::

    python scripts/forecast.py --scenarios config/scenarios.example.yaml

The weather basis defaults to ``inseason`` (real current-year ECCC weather to date + an
analog year for the rest of the season). Use ``--weather analog`` to forecast under a whole
historical season, or ``--weather synthetic`` to run offline.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import yaml

from canola_dt.config import load_config
from canola_dt.scenario import Scenario, run_scenario


def _build_single(args) -> Scenario:
    return Scenario(
        crop=args.crop, name=args.name or f"{args.crop}-scenario", province=args.province,
        station_id=args.station, weather=args.weather, analog_year=args.analog_year,
        seeding_date=date.fromisoformat(args.seeding_date) if args.seeding_date else None,
        preceding_crop=args.preceding or "", variety=args.variety or "",
        plants_per_m2=args.plants, n=args.n, p2o5=args.p2o5, k2o=args.k2o, s=args.s,
        soil_n=args.soil_n, soil_p2o5=args.soil_p2o5, soil_k2o=args.soil_k2o, soil_s=args.soil_s,
    )


def _load_scenarios(path: str) -> list[Scenario]:
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    if isinstance(doc, list):
        items, defaults = doc, {}
    else:
        items, defaults = doc.get("scenarios", []), doc.get("defaults", {})
    return [Scenario.from_dict({**defaults, **item}) for item in items]


def _print_one(r: dict) -> None:
    print(f"\n=== Forecast: {r['name']}  ({r['crop']}) ===")
    print(f"  weather       : {r['weather']}")
    lim = r["limiting_factor"]
    print(f"  YIELD         : {r['yield_t_ha']} t/ha  ({r['yield_bu_ac']} bu/ac)   [limited by: {lim}]")
    print(f"    biophysical : {r['biophysical_t_ha']} t/ha (water/weather potential)")
    if r["protein_pct"] is not None:
        print(f"  protein       : {r['protein_pct']} %")
    flower = "heading/anthesis" if r["crop"] == "wheat" else "flowering"
    print(f"  phenology     : {flower} day {r['days_to_flower']}, maturity day {r['days_to_maturity']} "
          f"(reached: {r['reached_maturity']})")
    f = r["fertilizer_kg_ha"]
    print(f"  fertility rec : N {f['N']}  P2O5 {f['P2O5']}  K2O {f['K2O']}  S {f['S']} kg/ha   "
          f"(limiting nutrient: {r['limiting_nutrient']})")
    if r["alerts"]:
        print("  planning alerts:")
        for a in r["alerts"]:
            print(f"    {a}")


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'name':<18} {'crop':<7} {'yield':>6} {'bu/ac':>6} {'prot':>5} "
          f"{'limited_by':<14} {'matur':>5} {'limN':>5}")
    print("-" * 74)
    for r in rows:
        prot = "-" if r["protein_pct"] is None else f"{r['protein_pct']:.1f}"
        print(f"{r['name'][:18]:<18} {r['crop']:<7} {r['yield_t_ha']:>6.2f} {r['yield_bu_ac']:>6.1f} "
              f"{prot:>5} {str(r['limiting_factor'])[:14]:<14} {str(r['days_to_maturity']):>5} "
              f"{str(r['limiting_nutrient']):>5}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Canola/wheat scenario forecasting")
    ap.add_argument("--scenarios", help="YAML file of scenarios for batch comparison")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    ap.add_argument("--crop", choices=["canola", "wheat"], default="wheat")
    ap.add_argument("--name", default="")
    ap.add_argument("--province", default="Saskatchewan")
    ap.add_argument("--station", type=int, default=None)
    ap.add_argument("--weather", choices=["inseason", "analog", "synthetic"], default="inseason")
    ap.add_argument("--analog-year", type=int, default=2022)
    ap.add_argument("--seeding-date", default="")
    ap.add_argument("--preceding", default="")
    ap.add_argument("--variety", default="")
    ap.add_argument("--plants", type=float, default=None, help="plants/m2 (density or population)")
    ap.add_argument("--n", type=float, default=None)
    ap.add_argument("--p2o5", type=float, default=40.0)
    ap.add_argument("--k2o", type=float, default=0.0)
    ap.add_argument("--s", type=float, default=None)
    ap.add_argument("--soil-n", type=float, default=30.0)
    ap.add_argument("--soil-p2o5", type=float, default=20.0)
    ap.add_argument("--soil-k2o", type=float, default=300.0)
    ap.add_argument("--soil-s", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.scenarios:
        scenarios = _load_scenarios(args.scenarios)
        print(f"Running {len(scenarios)} scenarios...")
        rows = [run_scenario(sc, cfg) for sc in scenarios]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            _print_table(rows)
            crit = [(r["name"], a) for r in rows for a in r["alerts"] if a.startswith("[CRITICAL]")]
            if crit:
                print("\nCRITICAL planning alerts:")
                for name, a in crit:
                    print(f"  {name}: {a}")
    else:
        r = run_scenario(_build_single(args), cfg)
        print(json.dumps(r, indent=2) if args.json else "", end="")
        if not args.json:
            _print_one(r)


if __name__ == "__main__":
    main()
