"""APSIM-style process-based spring-wheat crop model (pure Python).

Mirrors the canola process model (:mod:`canola_dt.simulation.process_model`) — same
radiation-use-efficiency biomass, layered cascading soil-water balance and FAO-56
agro-met primitives (:mod:`canola_dt.simulation.agromet`) — but with spring-wheat
phenology, canopy and harvest index:

* cardinal temperatures base 0 / opt 21 / max 35 °C (no vernalization — spring wheat);
* Zadoks-aligned stages: emergence -> tillering -> jointing -> heading -> anthesis ->
  grain fill -> maturity, advanced by thermal time with a long-day photoperiod modifier;
* higher harvest index (~0.42) than canola, reduced by heat and water stress during
  anthesis and grain fill.

Defaults are conventional Prairie spring-wheat values; calibrate against StatCan wheat
yields (see ``scripts/calibrate_wheat_model.py``) before operational use.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, replace
from enum import IntEnum

import numpy as np
import pandas as pd

from canola_dt.simulation import agromet

CALIBRATABLE = ("rue", "kl", "hi_heat_sensitivity", "harvest_index")


class WheatStage(IntEnum):
    """Ordered spring-wheat phenological stages (coarse, thermal-time driven)."""
    SOWN = 0
    EMERGENCE = 1
    TILLERING = 2
    JOINTING = 3       # stem elongation (Zadoks 30s)
    HEADING = 4        # Zadoks 50s
    ANTHESIS = 5       # Zadoks 60s (flowering; FHB / heat-sensitive)
    GRAIN_FILL = 6     # Zadoks 70s-80s
    MATURITY = 7       # Zadoks 90s


@dataclass
class WheatParameters:
    """Spring-wheat crop + soil parameters. Conventional Prairie defaults."""

    # --- Phenology: cardinal temperatures (deg C); spring wheat, no vernalization ---
    t_base: float = 0.0
    t_opt: float = 21.0
    t_max: float = 35.0
    # Thermal-time durations of each phase (deg-day, base 0). Photoperiod modifies the
    # pre-heading (vegetative) phases.
    tt_emergence: float = 120.0        # sowing -> emergence
    tt_to_jointing: float = 400.0      # emergence -> stem elongation (incl. tillering)
    tt_to_heading: float = 350.0       # jointing -> heading
    tt_to_anthesis: float = 80.0       # heading -> anthesis
    tt_anthesis: float = 100.0         # anthesis duration
    tt_grain_fill: float = 600.0       # grain fill -> physiological maturity
    pp_base: float = 8.0
    pp_opt: float = 16.0
    photoperiod_sensitive: bool = True

    # --- Canopy / biomass ---
    rue: float = 1.40                  # g biomass per MJ intercepted PAR
    k_extinction: float = 0.45         # Beer's-law extinction coefficient (wheat)
    lai_growth_rate: float = 0.0090    # LAI per deg-day during vegetative expansion
    senescence_rate: float = 0.004     # green-LAI loss per deg-day after anthesis
    lai_max: float = 6.0

    # --- Soil water (layered) ---
    layer_thickness_mm: tuple[float, ...] = (150.0, 150.0, 300.0, 300.0, 300.0)
    dul_frac: float = 0.30
    ll_frac: float = 0.13
    sat_frac: float = 0.45
    init_available_frac: float = 0.80  # Prairie spring recharge
    kl: float = 0.10
    root_front_rate: float = 1.6       # mm root depth per deg-day
    max_root_depth_mm: float = 1100.0  # wheat roots slightly deeper than canola
    runoff_cn: float = 75.0

    # --- Yield ---
    harvest_index: float = 0.42
    hi_heat_sensitivity: float = 0.02  # HI fraction lost per grain-fill heat-stress day
    hi_water_sensitivity: float = 0.25 # HI fraction lost at full grain-fill water stress
    heat_threshold_c: float = 30.0     # daily tmax above which wheat grain fill suffers
    min_hi_fraction: float = 0.4

    def with_overrides(self, overrides: dict) -> "WheatParameters":
        known = {f.name for f in fields(self)}
        return replace(self, **{k: v for k, v in overrides.items() if k in known})

    @classmethod
    def from_config(cls, cfg) -> "WheatParameters":
        return cls().with_overrides(cfg.get("wheat_model", {}) or {})

    @classmethod
    def from_calibrated(cls, cfg) -> "WheatParameters":
        params = cls.from_config(cfg)
        path = cfg.path("artifacts") / "wheat_calibrated_params.json"
        if path.exists():
            params = params.with_overrides(json.loads(path.read_text()))
        return params


@dataclass
class WheatModelResult:
    daily: pd.DataFrame
    summary: dict


class WheatCropModel:
    """Daily-step APSIM-style spring-wheat simulator for a single field-season."""

    def __init__(self, params: WheatParameters | None = None):
        self.p = params or WheatParameters()
        th = np.asarray(self.p.layer_thickness_mm, dtype=float)
        self._thick = th
        self._depth_top = np.concatenate([[0.0], np.cumsum(th)[:-1]])
        self._ll = self.p.ll_frac * th
        self._dul = self.p.dul_frac * th
        self._sat = self.p.sat_frac * th
        self._air_dry = 0.5 * self._ll

    def _runoff(self, precip: float) -> float:
        s = 25400.0 / self.p.runoff_cn - 254.0
        ia = 0.2 * s
        return 0.0 if precip <= ia else (precip - ia) ** 2 / (precip + 0.8 * s)

    def _root_fraction(self, root_depth: float) -> np.ndarray:
        frac = (root_depth - self._depth_top) / self._thick
        return np.clip(frac, 0.0, 1.0)

    def _phase_bounds(self) -> dict[WheatStage, float]:
        p = self.p
        r_joint = p.tt_to_jointing
        r_head = r_joint + p.tt_to_heading
        r_anth = r_head + p.tt_to_anthesis
        r_gf = r_anth + p.tt_anthesis
        r_mat = r_gf + p.tt_grain_fill
        return {
            WheatStage.TILLERING: 0.0,
            WheatStage.JOINTING: r_joint,
            WheatStage.HEADING: r_head,
            WheatStage.ANTHESIS: r_anth,
            WheatStage.GRAIN_FILL: r_gf,
            WheatStage.MATURITY: r_mat,
        }

    def _stage_for_dvt(self, dvt: float) -> WheatStage:
        stage = WheatStage.EMERGENCE
        for s, start in self._phase_bounds().items():
            if dvt >= start:
                stage = s
        return stage

    def run(self, weather: pd.DataFrame, latitude_deg: float) -> WheatModelResult:
        p = self.p
        w = weather.reset_index(drop=True)
        bounds = self._phase_bounds()
        heading_tt = bounds[WheatStage.HEADING]  # canopy expands until heading

        sw = self._ll + p.init_available_frac * (self._dul - self._ll)
        tt_sow = 0.0
        dvt = 0.0
        emerged = False
        biomass = 0.0
        lai = 0.0
        root_depth = 0.0

        grain_fill_heat_days = 0
        grain_fill_water_stress: list[float] = []
        records: list[dict] = []
        maturity_idx = None

        for i, row in w.iterrows():
            doy = int(pd.Timestamp(row["date"]).dayofyear)
            tmax, tmin = float(row["tmax_c"]), float(row["tmin_c"])
            tmean = float(row["tmean_c"])
            precip = float(row["precip_mm"]) if not math.isnan(row["precip_mm"]) else 0.0

            tt_day = agromet.thermal_time(tmean, p.t_base, p.t_opt, p.t_max)
            et0 = agromet.hargreaves_et0(doy, latitude_deg, tmax, tmin, tmean)

            stage = WheatStage.SOWN
            if not emerged:
                tt_sow += tt_day
                if tt_sow >= p.tt_emergence:
                    emerged = True
                    biomass = 1.0
                    lai = 0.05
                    root_depth = 50.0
                    stage = WheatStage.EMERGENCE
            else:
                if p.photoperiod_sensitive and dvt < heading_tt:
                    n = agromet.daylength(doy, latitude_deg)
                    dev = tt_day * agromet.photoperiod_factor(n, p.pp_base, p.pp_opt)
                else:
                    dev = tt_day
                dvt += dev
                stage = self._stage_for_dvt(dvt)
                if stage == WheatStage.MATURITY and maturity_idx is None:
                    maturity_idx = i

            # soil water: infiltration, runoff, drainage cascade
            runoff = self._runoff(precip)
            infil = precip - runoff
            sw[0] += infil
            sumes_reset = infil > 5.0
            for L in range(len(sw)):
                excess = sw[L] - self._dul[L]
                if excess > 0:
                    sw[L] = self._dul[L]
                    if L + 1 < len(sw):
                        sw[L + 1] += excess

            # soil evaporation (cover-limited, wetness-reduced)
            pot_soil_evap = et0 * math.exp(-p.k_extinction * lai)
            top_wet = (sw[0] - self._air_dry[0]) / max(1e-6, self._dul[0] - self._air_dry[0])
            soil_evap = max(0.0, min(pot_soil_evap * max(0.0, min(1.0, top_wet)),
                                     sw[0] - self._air_dry[0]))
            sw[0] -= soil_evap

            # transpiration demand, supply, water stress
            transp = 0.0
            swdef_photo = 1.0
            if emerged and lai > 0:
                root_frac = self._root_fraction(root_depth)
                avail = np.maximum(0.0, sw - self._ll) * root_frac
                supply = p.kl * avail
                total_supply = float(supply.sum())
                pot_transp = et0 * (1.0 - math.exp(-p.k_extinction * lai))
                transp = min(pot_transp, total_supply)
                if pot_transp > 0:
                    swdef_photo = max(0.0, min(1.0, total_supply / pot_transp))
                if total_supply > 0 and transp > 0:
                    sw -= transp * (supply / total_supply)
                    sw = np.maximum(sw, self._air_dry)

            water_stress = 1.0 - swdef_photo

            # biomass accumulation (RUE on intercepted PAR)
            if emerged:
                rs = agromet.solar_radiation(doy, latitude_deg, tmax, tmin)
                par = agromet.PAR_FRACTION * rs
                fint = 1.0 - math.exp(-p.k_extinction * lai)
                dW = p.rue * par * fint * swdef_photo
                biomass += dW

                if dvt < heading_tt:
                    lai = min(p.lai_max, lai + p.lai_growth_rate * tt_day * swdef_photo)
                if stage >= WheatStage.GRAIN_FILL:
                    lai = max(0.0, lai * (1.0 - p.senescence_rate * tt_day))
                if stage < WheatStage.MATURITY:
                    root_depth = min(p.max_root_depth_mm, root_depth + p.root_front_rate * tt_day)

                if stage in (WheatStage.ANTHESIS, WheatStage.GRAIN_FILL):
                    if tmax > p.heat_threshold_c:
                        grain_fill_heat_days += 1
                    grain_fill_water_stress.append(water_stress)

            records.append({
                "date": row["date"], "doy": doy, "stage": stage,
                "tt_cum": round(dvt, 1), "lai": round(lai, 3),
                "biomass_g_m2": round(biomass, 1), "root_depth_mm": round(root_depth, 0),
                "profile_paw_mm": round(float(np.maximum(0.0, sw - self._ll).sum()), 1),
                "et0_mm": round(et0, 2), "transp_mm": round(transp, 2),
                "soil_evap_mm": round(soil_evap, 2), "runoff_mm": round(runoff, 2),
                "water_stress": round(water_stress, 3),
            })

            if maturity_idx is not None:
                break

        daily = pd.DataFrame(records)
        mean_gf_ws = float(np.mean(grain_fill_water_stress)) if grain_fill_water_stress else 0.0
        hi = p.harvest_index * (
            1.0 - p.hi_heat_sensitivity * grain_fill_heat_days - p.hi_water_sensitivity * mean_gf_ws
        )
        hi = max(hi, p.min_hi_fraction * p.harvest_index)
        yield_kg_ha = biomass * hi * 10.0

        timeline = self._stage_dates(daily)
        summary = {
            "yield_kg_ha": round(yield_kg_ha, 1),
            "total_biomass_g_m2": round(biomass, 1),
            "harvest_index": round(hi, 3),
            "max_lai": round(float(daily["lai"].max()), 2),
            "reached_maturity": maturity_idx is not None,
            "days_to_heading": timeline.get(WheatStage.HEADING),
            "days_to_anthesis": timeline.get(WheatStage.ANTHESIS),
            "days_to_maturity": timeline.get(WheatStage.MATURITY),
            "grain_fill_heat_days": grain_fill_heat_days,
            "mean_grain_fill_water_stress": round(mean_gf_ws, 3),
            "season_transp_mm": round(float(daily["transp_mm"].sum()), 1),
            "season_soil_evap_mm": round(float(daily["soil_evap_mm"].sum()), 1),
        }
        return WheatModelResult(daily=daily, summary=summary)

    @staticmethod
    def _stage_dates(daily: pd.DataFrame) -> dict[WheatStage, int]:
        if daily.empty:
            return {}
        start = daily["date"].iloc[0]
        out: dict[WheatStage, int] = {}
        for stage, grp in daily.groupby("stage"):
            out[WheatStage(int(stage))] = int((grp["date"].iloc[0] - start).days)
        return out
