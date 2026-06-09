"""APSIM-style process-based yellow-pea crop model (pure Python).

Mirrors the cereal/canola process models (RUE biomass, layered soil-water balance, FAO-56
agro-met primitives) with yellow-pea (cool-season legume) phenology and a low flowering
heat-abort threshold:

* cardinal temperatures base 5 / opt 18 / max 30 °C (cool-season);
* stages emergence -> vegetative -> flowering -> pod fill -> maturity, ~85-100 days;
* peas are very heat-sensitive: flowers abort above ~25 °C, so harvest index is cut by heat
  and water stress through flowering and pod fill.

Nitrogen is NOT a biophysical input here — peas fix their own N (see the fertility/advisory
layers). Defaults are conventional Prairie values; calibrate against StatCan "Peas, dry"
(see ``scripts/calibrate_pea_model.py``) before operational use.
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


class PeaStage(IntEnum):
    """Ordered yellow-pea phenological stages (coarse, thermal-time driven)."""
    SOWN = 0
    EMERGENCE = 1
    VEGETATIVE = 2
    FLOWERING = 3      # heat-sensitive (flower abort > ~25 C)
    POD_FILL = 4
    MATURITY = 5


@dataclass
class PeaParameters:
    """Yellow-pea crop + soil parameters. Conventional Prairie defaults."""

    # --- Phenology: cardinal temperatures (deg C); cool-season legume ---
    t_base: float = 5.0
    t_opt: float = 18.0
    t_max: float = 30.0
    tt_emergence: float = 90.0
    tt_to_flowering: float = 400.0     # emergence -> flowering (~40-55 days)
    tt_flowering: float = 120.0        # flowering duration (heat-abort window)
    tt_pod_fill: float = 380.0         # pod fill -> physiological maturity (~85-100 day season)
    pp_base: float = 8.0
    pp_opt: float = 16.0
    photoperiod_sensitive: bool = True

    # --- Canopy / biomass ---
    rue: float = 1.20                  # legumes lower than cereals
    k_extinction: float = 0.50
    lai_growth_rate: float = 0.0090
    senescence_rate: float = 0.005
    lai_max: float = 5.0

    # --- Soil water (layered) ---
    layer_thickness_mm: tuple[float, ...] = (150.0, 150.0, 300.0, 300.0, 300.0)
    dul_frac: float = 0.30
    ll_frac: float = 0.13
    sat_frac: float = 0.45
    init_available_frac: float = 0.80
    kl: float = 0.10
    root_front_rate: float = 1.4       # peas root shallower than cereals
    max_root_depth_mm: float = 900.0
    runoff_cn: float = 75.0

    # --- Yield ---
    harvest_index: float = 0.48
    hi_heat_sensitivity: float = 0.03
    hi_water_sensitivity: float = 0.25
    heat_threshold_c: float = 25.0     # flower abort above this (LOW vs cereals)
    min_hi_fraction: float = 0.4

    def with_overrides(self, overrides: dict) -> "PeaParameters":
        known = {f.name for f in fields(self)}
        return replace(self, **{k: v for k, v in overrides.items() if k in known})

    @classmethod
    def from_config(cls, cfg) -> "PeaParameters":
        return cls().with_overrides(cfg.get("pea_model", {}) or {})

    @classmethod
    def from_calibrated(cls, cfg) -> "PeaParameters":
        params = cls.from_config(cfg)
        path = cfg.path("artifacts") / "pea_calibrated_params.json"
        if path.exists():
            params = params.with_overrides(json.loads(path.read_text()))
        return params


@dataclass
class PeaModelResult:
    daily: pd.DataFrame
    summary: dict


class PeaCropModel:
    """Daily-step APSIM-style yellow-pea simulator for a single field-season."""

    def __init__(self, params: PeaParameters | None = None):
        self.p = params or PeaParameters()
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

    def _phase_bounds(self) -> dict[PeaStage, float]:
        p = self.p
        r_flower = p.tt_to_flowering
        r_pod = r_flower + p.tt_flowering
        r_mat = r_pod + p.tt_pod_fill
        return {PeaStage.VEGETATIVE: 0.0, PeaStage.FLOWERING: r_flower,
                PeaStage.POD_FILL: r_pod, PeaStage.MATURITY: r_mat}

    def _stage_for_dvt(self, dvt: float) -> PeaStage:
        stage = PeaStage.EMERGENCE
        for s, start in self._phase_bounds().items():
            if dvt >= start:
                stage = s
        return stage

    def run(self, weather: pd.DataFrame, latitude_deg: float) -> PeaModelResult:
        p = self.p
        w = weather.reset_index(drop=True)
        flower_tt = self._phase_bounds()[PeaStage.FLOWERING]  # canopy expands to flowering

        sw = self._ll + p.init_available_frac * (self._dul - self._ll)
        tt_sow = 0.0
        dvt = 0.0
        emerged = False
        biomass = 0.0
        lai = 0.0
        root_depth = 0.0

        repro_heat_days = 0
        repro_water_stress: list[float] = []
        records: list[dict] = []
        maturity_idx = None

        for i, row in w.iterrows():
            doy = int(pd.Timestamp(row["date"]).dayofyear)
            tmax, tmin = float(row["tmax_c"]), float(row["tmin_c"])
            tmean = float(row["tmean_c"])
            precip = float(row["precip_mm"]) if not math.isnan(row["precip_mm"]) else 0.0

            tt_day = agromet.thermal_time(tmean, p.t_base, p.t_opt, p.t_max)
            et0 = agromet.hargreaves_et0(doy, latitude_deg, tmax, tmin, tmean)

            stage = PeaStage.SOWN
            if not emerged:
                tt_sow += tt_day
                if tt_sow >= p.tt_emergence:
                    emerged = True
                    biomass, lai, root_depth = 1.0, 0.05, 50.0
                    stage = PeaStage.EMERGENCE
            else:
                if p.photoperiod_sensitive and dvt < flower_tt:
                    n = agromet.daylength(doy, latitude_deg)
                    dev = tt_day * agromet.photoperiod_factor(n, p.pp_base, p.pp_opt)
                else:
                    dev = tt_day
                dvt += dev
                stage = self._stage_for_dvt(dvt)
                if stage == PeaStage.MATURITY and maturity_idx is None:
                    maturity_idx = i

            runoff = self._runoff(precip)
            sw[0] += precip - runoff
            for L in range(len(sw)):
                excess = sw[L] - self._dul[L]
                if excess > 0:
                    sw[L] = self._dul[L]
                    if L + 1 < len(sw):
                        sw[L + 1] += excess

            pot_soil_evap = et0 * math.exp(-p.k_extinction * lai)
            top_wet = (sw[0] - self._air_dry[0]) / max(1e-6, self._dul[0] - self._air_dry[0])
            soil_evap = max(0.0, min(pot_soil_evap * max(0.0, min(1.0, top_wet)),
                                     sw[0] - self._air_dry[0]))
            sw[0] -= soil_evap

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

            if emerged:
                rs = agromet.solar_radiation(doy, latitude_deg, tmax, tmin)
                par = agromet.PAR_FRACTION * rs
                fint = 1.0 - math.exp(-p.k_extinction * lai)
                biomass += p.rue * par * fint * swdef_photo

                if dvt < flower_tt:
                    lai = min(p.lai_max, lai + p.lai_growth_rate * tt_day * swdef_photo)
                if stage >= PeaStage.POD_FILL:
                    lai = max(0.0, lai * (1.0 - p.senescence_rate * tt_day))
                if stage < PeaStage.MATURITY:
                    root_depth = min(p.max_root_depth_mm, root_depth + p.root_front_rate * tt_day)

                if stage in (PeaStage.FLOWERING, PeaStage.POD_FILL):
                    repro_water_stress.append(water_stress)
                    # Heat counted only during the (short) flowering window -- peas abort
                    # flowers above ~25 C; counting the whole pod-fill window over-penalizes.
                    if stage == PeaStage.FLOWERING and tmax > p.heat_threshold_c:
                        repro_heat_days += 1

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
        mean_ws = float(np.mean(repro_water_stress)) if repro_water_stress else 0.0
        hi = p.harvest_index * (1.0 - p.hi_heat_sensitivity * repro_heat_days
                                - p.hi_water_sensitivity * mean_ws)
        hi = max(hi, p.min_hi_fraction * p.harvest_index)
        yield_kg_ha = biomass * hi * 10.0

        timeline = self._stage_dates(daily)
        summary = {
            "yield_kg_ha": round(yield_kg_ha, 1),
            "total_biomass_g_m2": round(biomass, 1),
            "harvest_index": round(hi, 3),
            "max_lai": round(float(daily["lai"].max()), 2),
            "reached_maturity": maturity_idx is not None,
            "days_to_flowering": timeline.get(PeaStage.FLOWERING),
            "days_to_maturity": timeline.get(PeaStage.MATURITY),
            "flowering_heat_days": repro_heat_days,
            "mean_flowering_water_stress": round(mean_ws, 3),
            "season_transp_mm": round(float(daily["transp_mm"].sum()), 1),
            "season_soil_evap_mm": round(float(daily["soil_evap_mm"].sum()), 1),
        }
        return PeaModelResult(daily=daily, summary=summary)

    @staticmethod
    def _stage_dates(daily: pd.DataFrame) -> dict[PeaStage, int]:
        if daily.empty:
            return {}
        start = daily["date"].iloc[0]
        out: dict[PeaStage, int] = {}
        for stage, grp in daily.groupby("stage"):
            out[PeaStage(int(stage))] = int((grp["date"].iloc[0] - start).days)
        return out
