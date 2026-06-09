"""Scenario forecasting for canola and wheat — a what-if layer over the digital twin.

A :class:`Scenario` bundles a crop, a weather basis and a management plan; ``run_scenario``
returns a structured forecast (yield, protein for wheat, limiting factor, phenology,
fertility recommendation and planning alerts) by driving the calibrated process model and
the advisory engine. ``compare_scenarios`` runs several and tabulates them.

Weather basis (for forecasting the *current* season):
  * ``inseason`` — real ECCC weather for the current year up to today, then an analog year
    for the remainder of the season (a genuine in-season forecast);
  * ``analog``   — a whole historical season used as the weather;
  * ``synthetic``— offline synthetic weather.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from canola_dt.advisory.engine import CanolaAdvisoryEngine
from canola_dt.advisory.state import CanolaFieldState
from canola_dt.advisory.agronomy import CultivarType, PrecedingCrop, Species
from canola_dt.advisory.wheat_agronomy import WheatClass, WheatPrecedingCrop
from canola_dt.advisory.wheat_engine import WheatAdvisoryEngine
from canola_dt.advisory.wheat_state import WheatFieldState
from canola_dt.advisory.barley_agronomy import BarleyPrecedingCrop, BarleyType
from canola_dt.advisory.barley_engine import BarleyAdvisoryEngine
from canola_dt.advisory.barley_state import BarleyFieldState
from canola_dt.advisory.pea_agronomy import PeaPrecedingCrop, PeaType
from canola_dt.advisory.pea_engine import PeaAdvisoryEngine
from canola_dt.advisory.pea_state import PeaFieldState
from canola_dt.config import load_config
from canola_dt.data import eccc
from canola_dt.data.ingest import synthetic_weather

SEASON_START = (5, 1)
SEASON_END = (10, 31)


@dataclass
class Scenario:
    """A what-if forecast configuration for one crop-season."""
    crop: str = "wheat"                      # "canola" | "wheat"
    name: str = "scenario"
    province: str = "Saskatchewan"
    station_id: int | None = None            # default: first station of the province
    weather: str = "inseason"                # inseason | analog | synthetic
    analog_year: int = 2022                  # weather for the remainder / analog season
    seeding_date: date | None = None         # default: May 5 of the current year

    # Management (None -> crop default)
    preceding_crop: str = ""
    variety: str = ""                        # cultivar (canola) / class (wheat)
    plants_per_m2: float | None = None       # density (canola) / population (wheat)
    n: float | None = None
    p2o5: float = 40.0
    k2o: float = 0.0
    s: float | None = None
    soil_n: float = 30.0
    soil_p2o5: float = 20.0
    soil_k2o: float = 300.0
    soil_s: float | None = None

    # Friendly aliases accepted in scenario files.
    _ALIASES = {"preceding": "preceding_crop", "plants": "plants_per_m2",
                "station": "station_id", "analog": "analog_year"}

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d = {cls._ALIASES.get(k, k): v for k, v in d.items()}
        if d.get("seeding_date"):
            d["seeding_date"] = date.fromisoformat(str(d["seeding_date"]))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        return cls(**d)


# --- weather assembly --------------------------------------------------------

def _prep_season(daily: pd.DataFrame, year: int) -> pd.DataFrame:
    s = daily[(daily["date"] >= pd.Timestamp(year, *SEASON_START))
              & (daily["date"] <= pd.Timestamp(year, *SEASON_END))].set_index("date").asfreq("D")
    s[["tmin_c", "tmax_c", "tmean_c"]] = s[["tmin_c", "tmax_c", "tmean_c"]].interpolate(limit=7)
    s["precip_mm"] = s["precip_mm"].fillna(0.0)
    return s.reset_index()


def build_weather(cfg, sc: Scenario, today: date | None = None):
    """Return (weather_frame, latitude, label) for the scenario's weather basis."""
    today = pd.Timestamp(today or date.today())
    stations = eccc.station_map(cfg)
    if sc.weather == "synthetic":
        lat = 51.5
        return synthetic_weather(year=sc.analog_year, n_days=165), lat, "synthetic weather"

    if sc.station_id and int(sc.station_id) in stations:
        sid, info = int(sc.station_id), stations[int(sc.station_id)]
    else:
        sid, info = next((s, i) for s, i in stations.items() if i["province"] == sc.province)
    lat = float(info["lat"])
    cache = cfg.path("data_raw") / "eccc"

    if sc.weather == "inseason":
        real = _prep_season(eccc.fetch_daily(sid, today.year, cache), today.year)
        real = real[real["date"] <= today].dropna(subset=["tmean_c"])
        if len(real) >= 15:
            analog = _prep_season(eccc.fetch_daily(sid, sc.analog_year, cache), sc.analog_year)
            analog["date"] = analog["date"].apply(lambda d: d.replace(year=today.year))
            fill = analog[analog["date"] > real["date"].max()]
            combined = pd.concat([real, fill]).reset_index(drop=True)
            label = (f"{info['name']}: {today.year} actual to {real['date'].max().date()} "
                     f"+ {sc.analog_year} analog")
            return combined, lat, label
        # not enough current-year data -> fall back to analog

    yr = sc.analog_year
    return _prep_season(eccc.fetch_daily(sid, yr, cache), yr), lat, f"{info['name']}: {yr} (analog)"


# --- enum resolution ---------------------------------------------------------

def _canola_preceding(value: str) -> PrecedingCrop:
    try:
        return PrecedingCrop(value)
    except ValueError:
        return PrecedingCrop.WHEAT


def _wheat_preceding(value: str) -> WheatPrecedingCrop:
    try:
        return WheatPrecedingCrop(value)
    except ValueError:
        return WheatPrecedingCrop.CANOLA


def _barley_preceding(value: str) -> BarleyPrecedingCrop:
    try:
        return BarleyPrecedingCrop(value)
    except ValueError:
        return BarleyPrecedingCrop.CANOLA


def _pea_preceding(value: str) -> PeaPrecedingCrop:
    try:
        return PeaPrecedingCrop(value)
    except ValueError:
        return PeaPrecedingCrop.CEREAL


# --- scenario execution ------------------------------------------------------

def run_scenario(sc: Scenario, cfg=None, today: date | None = None) -> dict:
    """Run one scenario and return a structured forecast dict."""
    cfg = cfg or load_config()
    weather, lat, wlabel = build_weather(cfg, sc, today)
    seeding = sc.seeding_date or date((today or date.today()).year, 5, 5)

    if sc.crop == "canola":
        engine = CanolaAdvisoryEngine.with_calibrated_model(cfg)
        state = CanolaFieldState(
            field_id=sc.name, species=Species.B_NAPUS,
            cultivar_type=_enum_or(CultivarType, sc.variety, CultivarType.HYBRID),
            seeding_date=seeding, preceding_crop=_canola_preceding(sc.preceding_crop),
            latitude=lat,
            n_applied_kg_per_ha=_d(sc.n, 150.0), p2o5_applied_kg_per_ha=sc.p2o5,
            s_applied_kg_per_ha=_d(sc.s, 15.0), k2o_applied_kg_per_ha=sc.k2o,
            soil_available_n_kg_per_ha=sc.soil_n, soil_available_p2o5_kg_per_ha=sc.soil_p2o5,
            soil_available_k2o_kg_per_ha=sc.soil_k2o, soil_available_s_kg_per_ha=_d(sc.soil_s, 10.0),
        )
        state.plant_density_per_m2 = _d(sc.plants_per_m2, 65.0)
    elif sc.crop == "wheat":
        engine = WheatAdvisoryEngine.with_calibrated_model(cfg)
        state = WheatFieldState(
            field_id=sc.name, wheat_class=_enum_or(WheatClass, sc.variety, WheatClass.CWRS),
            seeding_date=seeding, preceding_crop=_wheat_preceding(sc.preceding_crop),
            latitude=lat,
            n_applied_kg_per_ha=_d(sc.n, 110.0), p2o5_applied_kg_per_ha=sc.p2o5,
            s_applied_kg_per_ha=_d(sc.s, 12.0), k2o_applied_kg_per_ha=sc.k2o,
            soil_available_n_kg_per_ha=sc.soil_n, soil_available_p2o5_kg_per_ha=sc.soil_p2o5,
            soil_available_k2o_kg_per_ha=sc.soil_k2o, soil_available_s_kg_per_ha=_d(sc.soil_s, 8.0),
        )
        state.plant_population_per_m2 = _d(sc.plants_per_m2, 275.0)
    elif sc.crop == "barley":
        engine = BarleyAdvisoryEngine.with_calibrated_model(cfg)
        state = BarleyFieldState(
            field_id=sc.name, barley_type=_enum_or(BarleyType, sc.variety, BarleyType.MALT_2ROW),
            seeding_date=seeding, preceding_crop=_barley_preceding(sc.preceding_crop),
            latitude=lat,
            n_applied_kg_per_ha=_d(sc.n, 90.0), p2o5_applied_kg_per_ha=sc.p2o5,
            s_applied_kg_per_ha=_d(sc.s, 12.0), k2o_applied_kg_per_ha=sc.k2o,
            soil_available_n_kg_per_ha=sc.soil_n, soil_available_p2o5_kg_per_ha=sc.soil_p2o5,
            soil_available_k2o_kg_per_ha=sc.soil_k2o, soil_available_s_kg_per_ha=_d(sc.soil_s, 8.0),
        )
        state.plant_population_per_m2 = _d(sc.plants_per_m2, 250.0)
    elif sc.crop == "pea":
        engine = PeaAdvisoryEngine.with_calibrated_model(cfg)
        state = PeaFieldState(
            field_id=sc.name, pea_type=_enum_or(PeaType, sc.variety, PeaType.YELLOW),
            seeding_date=seeding, preceding_crop=_pea_preceding(sc.preceding_crop),
            latitude=lat,
            n_applied_kg_per_ha=_d(sc.n, 12.0), p2o5_applied_kg_per_ha=sc.p2o5,
            s_applied_kg_per_ha=_d(sc.s, 8.0), k2o_applied_kg_per_ha=sc.k2o,
            soil_available_n_kg_per_ha=sc.soil_n, soil_available_p2o5_kg_per_ha=sc.soil_p2o5,
            soil_available_k2o_kg_per_ha=sc.soil_k2o, soil_available_s_kg_per_ha=_d(sc.soil_s, 8.0),
        )
        state.plant_population_per_m2 = _d(sc.plants_per_m2, 80.0)
    else:
        raise ValueError(f"unknown crop: {sc.crop!r} (expected canola | wheat | barley | pea)")

    phen = engine.crop_model.run(weather, lat).summary
    engine.update_yield(state, weather, lat)
    state.day_of_season = 3  # planning-time alerts (rotation, stand, N)
    alerts, _ = engine.step(state)
    fert = engine.fertility_report(state, max(0.5, state.yield_potential_t_ha))

    stage_key = "days_to_anthesis" if sc.crop in ("wheat", "barley") else "days_to_flowering"
    return {
        "name": sc.name, "crop": sc.crop, "weather": wlabel,
        "yield_t_ha": state.yield_potential_t_ha, "yield_bu_ac": state.yield_potential_bu_ac,
        "protein_pct": (getattr(state, "estimated_protein_pct", None)
                        if sc.crop in ("wheat", "barley", "pea") else None),
        "limiting_factor": state.yield_breakdown.get("limiting_factor"),
        "biophysical_t_ha": round(state.yield_breakdown["biophysical_kg_ha"] / 1000, 2),
        "days_to_flower": phen.get(stage_key), "days_to_maturity": phen.get("days_to_maturity"),
        "reached_maturity": phen.get("reached_maturity"),
        "limiting_nutrient": fert["limiting_nutrient"],
        "fertilizer_kg_ha": fert["recommendation_kg_ha"],
        "malt_grade_ok": getattr(state, "malt_grade_ok", None) if sc.crop == "barley" else None,
        "alerts": [str(a) for a in alerts if a.severity.value in ("CRITICAL", "WARNING")],
    }


def compare_scenarios(scenarios: list[Scenario], cfg=None, today: date | None = None) -> list[dict]:
    cfg = cfg or load_config()
    return [run_scenario(sc, cfg, today) for sc in scenarios]


def _d(value, default):
    return default if value is None else value


def _enum_or(enum_cls, value, default):
    try:
        return enum_cls(value)
    except ValueError:
        return default
