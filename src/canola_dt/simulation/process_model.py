"""APSIM-style process-based canola crop model (pure Python).

A daily-step mechanistic model of a canola crop, capturing the core processes that
APSIM/DSSAT represent, at a tractable level of detail:

* **Phenology** — cardinal-temperature thermal time, with a long-day photoperiod
  modifier on the pre-floral phase, advancing the crop through emergence, rosette,
  bolting, flowering, ripening and maturity.
* **Canopy** — leaf area built from biomass partitioning (specific leaf area) with
  post-flowering senescence; Beer's-law light interception.
* **Biomass** — radiation-use efficiency (RUE) on intercepted PAR, down-regulated by
  water stress.
* **Soil water** — a layered cascading bucket (SCS-CN runoff, tipping-bucket drainage,
  two-source evaporation/transpiration, root-front growth, per-layer uptake).
* **Yield** — above-ground biomass at maturity x a harvest index reduced by
  flowering heat stress and flowering-period water stress.

The model consumes the canonical daily weather frame (``date, tmin_c, tmax_c,
tmean_c, precip_mm``) plus site latitude. It is calibrated with conventional Prairie
canola defaults — treat parameters as starting points, not validated values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from canola_dt.constants import GrowthStage
from canola_dt.simulation import agromet

# Parameters exposed to the calibration routine.
CALIBRATABLE = ("rue", "kl", "hi_heat_sensitivity", "harvest_index")


@dataclass
class CanolaParameters:
    """Canola crop + soil parameters. Conventional Prairie defaults."""

    # --- Phenology: cardinal temperatures for thermal time (deg C) ---
    t_base: float = 5.0
    t_opt: float = 20.0
    t_max: float = 30.0
    # Thermal-time durations of each phase (deg-day). Photoperiod modifies the
    # emergence->floral-initiation phase only.
    tt_emergence: float = 100.0        # sowing -> emergence
    tt_to_floral_init: float = 300.0   # emergence -> floral initiation (rosette)
    tt_to_flowering: float = 300.0     # floral init -> first flower (bolting)
    tt_flowering: float = 300.0        # flowering duration
    tt_to_maturity: float = 400.0      # end of flowering -> physiological maturity
    # Photoperiod (long-day): development scaled in [pp_base, pp_opt] hours.
    pp_base: float = 10.0
    pp_opt: float = 16.0
    photoperiod_sensitive: bool = True

    # --- Canopy / biomass ---
    rue: float = 1.3                   # g biomass per MJ intercepted PAR
    k_extinction: float = 0.55         # Beer's-law extinction coefficient
    lai_growth_rate: float = 0.0085    # LAI per deg-day during vegetative expansion
    senescence_rate: float = 0.004     # green-LAI loss per deg-day after flowering
    lai_max: float = 5.5

    # --- Soil water (layered) ---
    layer_thickness_mm: tuple[float, ...] = (150.0, 150.0, 300.0, 300.0, 300.0)
    dul_frac: float = 0.30             # drained upper limit (volumetric)
    ll_frac: float = 0.13              # crop lower limit (volumetric)
    sat_frac: float = 0.45             # saturation (volumetric)
    init_available_frac: float = 0.80  # initial PAW fraction at sowing (Prairie spring recharge)
    kl: float = 0.10                   # per-layer root water-uptake coefficient (/day)
    root_front_rate: float = 1.5       # mm root depth per deg-day
    max_root_depth_mm: float = 1000.0
    runoff_cn: float = 75.0            # SCS curve number

    # --- Yield ---
    harvest_index: float = 0.28
    hi_heat_sensitivity: float = 0.03  # HI fraction lost per flowering heat-stress day
    hi_water_sensitivity: float = 0.25 # HI fraction lost at full flowering water stress
    heat_threshold_c: float = 29.5     # daily tmax above which canola flowers abort
    min_hi_fraction: float = 0.4       # floor on stressed HI relative to potential

    def with_overrides(self, overrides: dict) -> "CanolaParameters":
        """Return a copy with the given fields replaced (unknown keys ignored)."""
        known = {f.name for f in fields(self)}
        return replace(self, **{k: v for k, v in overrides.items() if k in known})

    @classmethod
    def from_config(cls, cfg) -> "CanolaParameters":
        """Build from the ``process_model`` section of config.yaml (over defaults)."""
        return cls().with_overrides(cfg.get("process_model", {}) or {})

    @classmethod
    def from_calibrated(cls, cfg) -> "CanolaParameters":
        """Like :meth:`from_config`, then overlay ``artifacts/calibrated_params.json``.

        Calibrated values (if the file exists) take precedence, so a calibration run
        immediately takes effect everywhere parameters are loaded this way.
        """
        params = cls.from_config(cfg)
        path = cfg.path("artifacts") / "calibrated_params.json"
        if path.exists():
            params = params.with_overrides(json.loads(path.read_text()))
        return params


@dataclass
class CropModelResult:
    daily: pd.DataFrame
    summary: dict


class CanolaCropModel:
    """Daily-step APSIM-style canola simulator for a single field-season."""

    def __init__(self, params: CanolaParameters | None = None):
        self.p = params or CanolaParameters()
        # Per-layer water-holding limits (mm), derived from volumetric fractions.
        th = np.asarray(self.p.layer_thickness_mm, dtype=float)
        self._thick = th
        self._depth_top = np.concatenate([[0.0], np.cumsum(th)[:-1]])
        self._ll = self.p.ll_frac * th
        self._dul = self.p.dul_frac * th
        self._sat = self.p.sat_frac * th
        self._air_dry = 0.5 * self._ll  # soil can dry below LL via evaporation

    # ---- soil-water helpers -------------------------------------------------

    def _runoff(self, precip: float) -> float:
        """SCS curve-number runoff (mm)."""
        s = 25400.0 / self.p.runoff_cn - 254.0
        ia = 0.2 * s
        if precip <= ia:
            return 0.0
        return (precip - ia) ** 2 / (precip + 0.8 * s)

    def _root_fraction(self, root_depth: float) -> np.ndarray:
        """Fraction of each layer occupied by roots, in [0, 1]."""
        bottom = self._depth_top + self._thick
        frac = (root_depth - self._depth_top) / self._thick
        return np.clip(frac, 0.0, 1.0)

    # ---- main loop ----------------------------------------------------------

    def run(self, weather: pd.DataFrame, latitude_deg: float) -> CropModelResult:
        p = self.p
        w = weather.reset_index(drop=True)

        # Initial soil water: init fraction of plant-available water above LL.
        sw = self._ll + p.init_available_frac * (self._dul - self._ll)

        tt_sow = 0.0          # thermal time since sowing (for emergence)
        dvt = 0.0             # development thermal time since emergence
        emerged = False
        biomass = 0.0         # above-ground dry matter, g m-2
        lai = 0.0
        root_depth = 0.0
        sumes = 0.0           # cumulative soil evaporation since last major wetting

        flowering_heat_days = 0
        flowering_water_stress: list[float] = []
        records: list[dict] = []
        maturity_idx = None

        for i, row in w.iterrows():
            doy = int(pd.Timestamp(row["date"]).dayofyear)
            tmax, tmin = float(row["tmax_c"]), float(row["tmin_c"])
            tmean = float(row["tmean_c"])
            precip = float(row["precip_mm"]) if not math.isnan(row["precip_mm"]) else 0.0

            tt_day = agromet.thermal_time(tmean, p.t_base, p.t_opt, p.t_max)
            et0 = agromet.hargreaves_et0(doy, latitude_deg, tmax, tmin, tmean)

            # --- phenology ---
            stage = GrowthStage.SOWN
            if not emerged:
                tt_sow += tt_day
                if tt_sow >= p.tt_emergence:
                    emerged = True
                    biomass = 1.0           # ~ emergence seedling biomass, g m-2
                    lai = 0.05
                    root_depth = 50.0
                    stage = GrowthStage.EMERGENCE
            else:
                # Photoperiod modifies development only pre-floral-initiation.
                if p.photoperiod_sensitive and dvt < p.tt_to_floral_init:
                    n = agromet.daylength(doy, latitude_deg)
                    dev = tt_day * agromet.photoperiod_factor(n, p.pp_base, p.pp_opt)
                else:
                    dev = tt_day
                dvt += dev
                stage = self._stage_for_dvt(dvt)
                if stage == GrowthStage.MATURITY and maturity_idx is None:
                    maturity_idx = i

            # --- soil water: infiltration, runoff, drainage cascade ---
            runoff = self._runoff(precip)
            infil = precip - runoff
            sw[0] += infil
            if infil > 5.0:
                sumes = 0.0  # reset evaporation stage after meaningful rain
            for L in range(len(sw)):
                excess = sw[L] - self._dul[L]
                if excess > 0:
                    sw[L] = self._dul[L]
                    if L + 1 < len(sw):
                        sw[L + 1] += excess
                    # else: deep drainage, lost from profile

            # --- soil evaporation (cover-limited, wetness-reduced) ---
            pot_soil_evap = et0 * math.exp(-p.k_extinction * lai)
            top_wet = (sw[0] - self._air_dry[0]) / max(1e-6, self._dul[0] - self._air_dry[0])
            soil_evap = max(0.0, min(pot_soil_evap * max(0.0, min(1.0, top_wet)),
                                     sw[0] - self._air_dry[0]))
            sw[0] -= soil_evap
            sumes += soil_evap

            # --- transpiration demand, supply, water stress ---
            transp = 0.0
            swdef_photo = 1.0
            if emerged and lai > 0:
                root_frac = self._root_fraction(root_depth)
                avail = np.maximum(0.0, sw - self._ll) * root_frac
                supply = p.kl * avail                       # mm per layer
                total_supply = float(supply.sum())
                pot_transp = et0 * (1.0 - math.exp(-p.k_extinction * lai))
                transp = min(pot_transp, total_supply)
                if pot_transp > 0:
                    swdef_photo = max(0.0, min(1.0, total_supply / pot_transp))
                if total_supply > 0 and transp > 0:
                    sw -= transp * (supply / total_supply)
                    sw = np.maximum(sw, self._air_dry)

            water_stress = 1.0 - swdef_photo

            # --- biomass accumulation (RUE on intercepted PAR) ---
            if emerged:
                rs = agromet.solar_radiation(doy, latitude_deg, tmax, tmin)
                par = agromet.PAR_FRACTION * rs
                fint = 1.0 - math.exp(-p.k_extinction * lai)
                dW = p.rue * par * fint * swdef_photo
                biomass += dW

                # Canopy expansion is thermal-time (sink) driven during vegetative
                # growth, reduced by water stress; it holds through flowering and
                # then senesces during ripening.
                pre_flower_tt = p.tt_to_floral_init + p.tt_to_flowering
                if dvt < pre_flower_tt:
                    lai = min(p.lai_max, lai + p.lai_growth_rate * tt_day * swdef_photo)
                if stage >= GrowthStage.RIPENING:
                    lai = max(0.0, lai * (1.0 - p.senescence_rate * tt_day))

                # Root front growth.
                if stage < GrowthStage.MATURITY:
                    root_depth = min(p.max_root_depth_mm, root_depth + p.root_front_rate * tt_day)

                # Track flowering-period stresses for harvest-index reduction.
                if stage == GrowthStage.FLOWERING:
                    if tmax > p.heat_threshold_c:
                        flowering_heat_days += 1
                    flowering_water_stress.append(water_stress)

            records.append({
                "date": row["date"],
                "doy": doy,
                "stage": stage,
                "tt_cum": round(dvt, 1),
                "lai": round(lai, 3),
                "biomass_g_m2": round(biomass, 1),
                "root_depth_mm": round(root_depth, 0),
                "profile_paw_mm": round(float(np.maximum(0.0, sw - self._ll).sum()), 1),
                "et0_mm": round(et0, 2),
                "transp_mm": round(transp, 2),
                "soil_evap_mm": round(soil_evap, 2),
                "runoff_mm": round(runoff, 2),
                "water_stress": round(water_stress, 3),
            })

            if maturity_idx is not None:
                break

        daily = pd.DataFrame(records)

        # --- yield from biomass x stress-reduced harvest index ---
        mean_flower_ws = float(np.mean(flowering_water_stress)) if flowering_water_stress else 0.0
        hi = p.harvest_index * (
            1.0
            - p.hi_heat_sensitivity * flowering_heat_days
            - p.hi_water_sensitivity * mean_flower_ws
        )
        hi = max(hi, p.min_hi_fraction * p.harvest_index)
        yield_kg_ha = biomass * hi * 10.0  # g m-2 -> kg ha-1

        timeline = self._stage_dates(daily)
        summary = {
            "yield_kg_ha": round(yield_kg_ha, 1),
            "total_biomass_g_m2": round(biomass, 1),
            "harvest_index": round(hi, 3),
            "max_lai": round(float(daily["lai"].max()), 2),
            "reached_maturity": maturity_idx is not None,
            "days_to_flowering": timeline.get(GrowthStage.FLOWERING),
            "days_to_maturity": timeline.get(GrowthStage.MATURITY),
            "flowering_heat_days": flowering_heat_days,
            "mean_flowering_water_stress": round(mean_flower_ws, 3),
            "season_transp_mm": round(float(daily["transp_mm"].sum()), 1),
            "season_soil_evap_mm": round(float(daily["soil_evap_mm"].sum()), 1),
        }
        return CropModelResult(daily=daily, summary=summary)

    # ---- phenology helpers --------------------------------------------------

    def _phase_bounds(self) -> dict[GrowthStage, float]:
        """Cumulative post-emergence thermal-time at the *start* of each stage."""
        p = self.p
        r1 = p.tt_to_floral_init
        r2 = r1 + p.tt_to_flowering
        r3 = r2 + p.tt_flowering
        r4 = r3 + p.tt_to_maturity
        return {
            GrowthStage.ROSETTE: 0.0,
            GrowthStage.BOLTING: r1,
            GrowthStage.FLOWERING: r2,
            GrowthStage.RIPENING: r3,
            GrowthStage.MATURITY: r4,
        }

    def _stage_for_dvt(self, dvt: float) -> GrowthStage:
        stage = GrowthStage.EMERGENCE
        for s, start in self._phase_bounds().items():
            if dvt >= start:
                stage = s
        return stage

    @staticmethod
    def _stage_dates(daily: pd.DataFrame) -> dict[GrowthStage, int]:
        """Days from start to first occurrence of each stage."""
        if daily.empty:
            return {}
        start = daily["date"].iloc[0]
        out: dict[GrowthStage, int] = {}
        for stage, grp in daily.groupby("stage"):
            out[GrowthStage(int(stage))] = int((grp["date"].iloc[0] - start).days)
        return out
