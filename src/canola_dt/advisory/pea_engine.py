"""Yellow-pea advisory engine (application layer).

The legume twist runs through everything: peas fix their own N, so the engine flags
**inoculation** and warns when applied N is high enough to *suppress* fixation (the opposite
of the cereal logic). It also handles the long pulse-free rotation that aphanomyces root rot
demands, ascochyta fungicide timing, the low 25 C flowering heat-abort, pea aphid / leaf
weevil, lodging and harvest/desiccation.

Yield = calibrated pea model x nutrient-limited ceiling (N met by fixation, so usually P or
water limits) x management modifiers (population, rotation). Protein (~22-25%) is estimated
with a mild yield-dilution model.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from canola_dt.advisory.agronomy import AlertSeverity
from canola_dt.advisory.state import Alert
from canola_dt.advisory.pea_agronomy import (
    PeaAgronomyParameters,
    PeaGrowthStage,
    PeaPrecedingCrop,
)
from canola_dt.advisory.pea_state import PeaFieldState
from canola_dt import fertility as fert
from canola_dt.data.aafc import PEA_BU_AC_TO_KG_HA
from canola_dt.simulation.pea_model import PeaCropModel, PeaParameters

_S = AlertSeverity


def _applied_nutrients(state: PeaFieldState) -> dict[str, float]:
    return {"N": state.n_applied_kg_per_ha, "P2O5": state.p2o5_applied_kg_per_ha,
            "K2O": state.k2o_applied_kg_per_ha, "S": state.s_applied_kg_per_ha}


def _fixation(state: PeaFieldState, p: PeaAgronomyParameters) -> float:
    """N fixed (kg/ha), reduced when applied N suppresses nodulation."""
    full = (p.N_fixation_kg_per_ha_min + p.N_fixation_kg_per_ha_max) / 2
    if state.n_applied_kg_per_ha >= p.N_fixation_prevented_kg_per_ha:
        return 0.0
    if state.n_applied_kg_per_ha >= p.N_nodulation_inhibit_kg_per_ha:
        return full * 0.5
    return full


def _soil_supply(state: PeaFieldState, p: PeaAgronomyParameters) -> dict[str, float]:
    # Add N fixation to the N supply so peas are not spuriously N-limited.
    return {"N": state.soil_available_n_kg_per_ha + _fixation(state, p),
            "P2O5": state.soil_available_p2o5_kg_per_ha,
            "K2O": state.soil_available_k2o_kg_per_ha, "S": state.soil_available_s_kg_per_ha}


class PeaAdvisoryEngine:
    """Decision-support engine for yellow peas; yield via the calibrated pea model."""

    def __init__(self, agronomy: PeaAgronomyParameters | None = None,
                 crop_model: PeaCropModel | None = None):
        self.params = agronomy or PeaAgronomyParameters()
        self.crop_model = crop_model or PeaCropModel(PeaParameters())

    @classmethod
    def with_calibrated_model(cls, cfg=None, agronomy: PeaAgronomyParameters | None = None):
        from canola_dt.config import load_config
        cfg = cfg or load_config()
        return cls(agronomy=agronomy, crop_model=PeaCropModel(PeaParameters.from_calibrated(cfg)))

    # ── Public API ───────────────────────────────────────────────────────────

    def step(self, state: PeaFieldState) -> tuple[list[Alert], list[str]]:
        self._update_growth_stage(state)
        alerts: list[Alert] = []
        alerts += self._check_inoculation(state)
        alerts += self._check_starter_n(state)
        alerts += self._check_rotation(state)
        alerts += self._check_population(state)
        alerts += self._check_weevil(state)
        alerts += self._check_aphids(state)
        alerts += self._check_ascochyta(state)
        alerts += self._check_heat(state)
        alerts += self._check_lodging(state)
        alerts += self._check_harvest(state)
        self._estimate_harvest_date(state)
        state.alert_log.extend(alerts)
        return alerts, [a.recommendation for a in alerts if a.recommendation]

    def update_yield(self, state: PeaFieldState, weather, latitude: float | None = None):
        lat = latitude if latitude is not None else state.latitude
        bio = self.crop_model.run(weather, lat).summary["yield_kg_ha"]
        nl = fert.nutrient_limited_yield(_applied_nutrients(state), _soil_supply(state, self.params),
                                         params=fert.pea_nutrient_parameters())
        nutrient_ceiling = nl.yield_t_ha * 1000.0
        attainable = min(bio, nutrient_ceiling)
        limiting = nl.limiting_nutrient if nutrient_ceiling < bio else None
        mods = self._management_modifiers(state)
        kg_ha = attainable * mods["combined"]
        state.yield_potential_t_ha = round(kg_ha / 1000.0, 2)
        state.yield_potential_bu_ac = round(kg_ha / PEA_BU_AC_TO_KG_HA, 1)
        state.estimated_protein_pct = self._estimate_protein(state)
        state.yield_breakdown = {
            "biophysical_kg_ha": round(bio, 1),
            "nutrient_ceiling_kg_ha": round(nutrient_ceiling, 1),
            "limiting_factor": limiting if limiting else "water/weather",
            "n_fixed_kg_ha": round(_fixation(state, self.params), 0),
            "population_mod": mods["population"], "rotation_mod": mods["rotation"],
            "final_kg_ha": round(kg_ha, 1), "estimated_protein_pct": state.estimated_protein_pct,
        }
        return state.yield_breakdown

    def run_season(self, state: PeaFieldState, sensor_readings: list[dict],
                   weather=None, latitude: float | None = None) -> dict[str, Any]:
        all_alerts: list[Alert] = []
        for reading in sensor_readings:
            state.ingest_sensor_reading(**reading)
            alerts, _ = self.step(state)
            all_alerts.extend(alerts)
        if weather is not None:
            self.update_yield(state, weather, latitude)
        return self._build_season_summary(state, all_alerts)

    def fertility_report(self, state: PeaFieldState, target_yield_t_ha: float) -> dict[str, Any]:
        """Pea fertility: N is fixed (starter only), P by removal, S light."""
        p = self.params
        pp = fert.pea_nutrient_parameters()
        removal = fert.seed_removal(target_yield_t_ha, pp)
        soil, applied = _soil_supply(state, p), _applied_nutrients(state)
        starter_n = round((p.starter_N_kg_per_ha_min + p.starter_N_kg_per_ha_max) / 2, 0)
        rec = {
            "N": starter_n,  # starter only — the crop fixes the rest
            "P2O5": round(max(0.0, removal["P2O5"] - state.soil_available_p2o5_kg_per_ha), 1),
            "K2O": 0.0,
            "S": round(max(0.0, removal["S"] + p.S_recommended_kg_per_ha
                           - state.soil_available_s_kg_per_ha), 1),
        }
        nl = fert.nutrient_limited_yield(applied, soil, pp)
        return {
            "target_yield_t_ha": target_yield_t_ha,
            "recommendation_kg_ha": rec,
            "n_fixed_kg_ha": round(_fixation(state, p), 0),
            "n_credit_to_next_crop_kg_per_ha": round(
                (p.N_credit_to_next_crop_kg_per_ha_min + p.N_credit_to_next_crop_kg_per_ha_max) / 2, 0),
            "nutrient_limited_yield_t_ha": nl.yield_t_ha,
            "limiting_nutrient": nl.limiting_nutrient,
            "note": "Peas fix N via rhizobia — apply only starter N; high N suppresses fixation.",
        }

    # ── Yield / protein helpers ───────────────────────────────────────────────

    def _management_modifiers(self, state: PeaFieldState) -> dict[str, float]:
        p = self.params
        pop = state.plant_population_per_m2
        if pop <= 0 or pop >= p.target_population_per_m2_min:
            pop_mod = 1.0
        elif pop >= p.thin_stand_per_m2:
            pop_mod = 0.95
        elif pop >= p.critical_stand_per_m2:
            pop_mod = 0.85
        else:
            pop_mod = 0.70
        rotation_mod = p.preceding_crop_yield_index.get(state.preceding_crop.value, 1.0)
        # No N modifier — peas fix their N.
        return {"population": round(pop_mod, 3), "rotation": round(rotation_mod, 3),
                "combined": round(pop_mod * rotation_mod, 4)}

    def _estimate_protein(self, state: PeaFieldState) -> float:
        """Pea protein (~22-25%): mild yield-dilution (higher yield -> slightly lower protein)."""
        p = self.params
        protein = 24.5 - 0.4 * state.yield_potential_t_ha
        return round(max(p.protein_pct_min, min(p.protein_pct_max, protein)), 1)

    # ── Phenology (day-count) ─────────────────────────────────────────────────

    def _update_growth_stage(self, state: PeaFieldState) -> None:
        dos = state.day_of_season
        G = PeaGrowthStage
        thresholds = [(7, G.GERMINATION), (10, G.EMERGENCE), (45, G.VEGETATIVE),
                      (58, G.FLOWERING), (95, G.POD_FILL)]
        state.growth_stage = G.MATURITY
        for day_max, stage in thresholds:
            if dos < day_max:
                state.growth_stage = stage
                break

    # ── Alert checks ──────────────────────────────────────────────────────────

    def _check_inoculation(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        if state.day_of_season > 8 or not p.inoculant_required:
            return []
        if not state.inoculant_applied:
            sev = (_S.CRITICAL if state.preceding_crop not in (PeaPrecedingCrop.PEA, PeaPrecedingCrop.PULSE)
                   else _S.WARNING)
            return [Alert(sev, "Inoculation",
                          "No rhizobium inoculant recorded. Peas fix most of their N via Rhizobium "
                          "leguminosarum — without effective nodulation, yield and protein suffer.",
                          "Inoculate at seeding (granular/liquid/peat). Essential on ground with no recent "
                          "pulse history; nodules become active 3-4 weeks after seeding.",
                          state.day_of_season)]
        return []

    def _check_starter_n(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        if state.day_of_season > 8:
            return []
        n = state.n_applied_kg_per_ha
        if n >= p.N_fixation_prevented_kg_per_ha:
            return [Alert(_S.CRITICAL, "NitrogenSuppressesFixation",
                          f"Applied N {n:.0f} kg/ha is at/above the level that PREVENTS fixation "
                          f"(~{p.N_fixation_prevented_kg_per_ha:.0f} kg/ha). The crop will rely on soil/fertilizer "
                          "N instead of fixing — costly and lower-yielding.",
                          f"Cut N to a small starter ({p.starter_N_kg_per_ha_min:.0f}-"
                          f"{p.starter_N_kg_per_ha_max:.0f} kg/ha). Let the rhizobia do the work.",
                          state.day_of_season)]
        if n > p.N_nodulation_inhibit_kg_per_ha:
            return [Alert(_S.WARNING, "NitrogenSuppressesFixation",
                          f"Applied N {n:.0f} kg/ha exceeds the nodulation-inhibition threshold "
                          f"(~{p.N_nodulation_inhibit_kg_per_ha:.0f} kg/ha) — it partly suppresses fixation.",
                          "Peas only need a small starter N; high N delays/reduces nodulation and wastes money.",
                          state.day_of_season)]
        return []

    def _check_rotation(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        if state.day_of_season > 8:
            return []
        if state.preceding_crop in (PeaPrecedingCrop.PEA, PeaPrecedingCrop.PULSE) \
                or state.years_since_last_pulse < p.rotation_min_years_from_pulse:
            sev = _S.CRITICAL if state.preceding_crop == PeaPrecedingCrop.PEA else _S.WARNING
            return [Alert(sev, "CropRotation",
                          f"Short pulse rotation ({state.preceding_crop.value}, "
                          f"{state.years_since_last_pulse} yr since pulse). Aphanomyces root rot has NO "
                          f"in-crop control and persists {p.aphanomyces_soil_persistence_years_min}-"
                          f"{p.aphanomyces_soil_persistence_years_max} years in soil.",
                          f"Keep {p.rotation_recommended_years_from_pulse}+ years between pulses; test soil for "
                          "aphanomyces before seeding peas; avoid wet/compacted fields.",
                          state.day_of_season)]
        return []

    def _check_population(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        pop = state.plant_population_per_m2
        if pop == 0 or pop >= p.target_population_per_m2_min:
            return []
        sev = _S.WARNING if pop >= p.critical_stand_per_m2 else _S.CRITICAL
        return [Alert(sev, "PlantPopulation",
                      f"Stand {pop:.0f} plants/m² below target ({p.target_population_per_m2_min}-"
                      f"{p.target_population_per_m2_max}). Thin pea stands compete poorly with weeds and "
                      "lodge more.",
                      "Aim for early, even weed control; peas don't compensate as well as cereals.",
                      state.day_of_season)]

    def _check_weevil(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage in (PeaGrowthStage.EMERGENCE, PeaGrowthStage.VEGETATIVE) \
                and state.weevil_damage_pct >= p.pea_leaf_weevil_damage_threshold_pct:
            return [Alert(_S.WARNING, "PeaLeafWeevil",
                          f"Pea leaf weevil feeding on {state.weevil_damage_pct:.0f}% of plants "
                          f"(threshold {p.pea_leaf_weevil_damage_threshold_pct:.0f}%). The damaging window is "
                          f"node {p.pea_leaf_weevil_window_node_start}-{p.pea_leaf_weevil_window_node_end}; "
                          "larvae feed on nodules and cut N fixation.",
                          "Treat at the seedling/clam-leaf stage if notching is widespread; a seed treatment "
                          "is the better long-term tool.", state.day_of_season)]
        return []

    def _check_aphids(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage in (PeaGrowthStage.FLOWERING, PeaGrowthStage.POD_FILL) \
                and state.aphids_per_tip >= p.pea_aphid_threshold_per_tip:
            return [Alert(_S.WARNING, "PeaAphid",
                          f"Pea aphids {state.aphids_per_tip:.0f} per 20 cm tip ≥ threshold "
                          f"({p.pea_aphid_threshold_per_tip:.0f}/tip) at flowering/pod fill.",
                          "Consider control; aphids are worst in warm, dry spells. Preserve beneficials where "
                          "possible.", state.day_of_season)]
        return []

    def _check_ascochyta(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage == PeaGrowthStage.FLOWERING \
                and (state.ascochyta_severity_pct > 5 or state.disease_pressure.get("ascochyta", 0.0) > 0.3
                     or state.soil_moisture_pct > 40):
            return [Alert(_S.WARNING, "AscochytaBlight",
                          f"Ascochyta risk at flowering (severity {state.ascochyta_severity_pct:.0f}%, "
                          f"soil moisture {state.soil_moisture_pct:.0f}%). It can cost up to "
                          f"{p.ascochyta_yield_loss_max_pct:.0f}% in wet years.",
                          f"Fungicide timing: {p.ascochyta_fungicide_timing}. Rotate and use clean seed.",
                          state.day_of_season)]
        return []

    def _check_heat(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage == PeaGrowthStage.FLOWERING \
                and state.air_temp_max_c > p.heat_abort_threshold_c:
            return [Alert(_S.WARNING, "HeatStress",
                          f"Air temp {state.air_temp_max_c:.0f}°C exceeds the flower-abort threshold "
                          f"({p.heat_abort_threshold_c:.0f}°C). Peas drop flowers and pods in heat — a key "
                          "yield risk in hot Prairie summers.",
                          "No in-field fix. Early seeding shifts flowering to cooler weather; note for the "
                          "yield outlook.", state.day_of_season)]
        return []

    def _check_lodging(self, state: PeaFieldState) -> list[Alert]:
        if state.lodging_pct <= 0:
            return []
        return [Alert(_S.WARNING, "Lodging",
                      f"Lodging {state.lodging_pct:.0f}% — peas vine and lodge, which slows dry-down, raises "
                      "disease and complicates harvest.",
                      "Use semi-leafless varieties and avoid excess seeding rate; harvest with lifters/low "
                      "cutterbar.", state.day_of_season)]

    def _check_harvest(self, state: PeaFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage != PeaGrowthStage.MATURITY:
            return []
        return [Alert(_S.INFO, "HarvestReadiness",
                      f"Maturity. Bottom pods ripening ({state.pod_brown_pct:.0f}% brown). "
                      f"Swath when ~{p.swath_bottom_pods_ripe_pct:.0f}% of bottom pods are ripe; desiccate "
                      f"(glyphosate) at ~{p.desiccation_pod_brown_pct:.0f}% pod brown / "
                      f"{p.desiccation_moisture_pct:.0f}% seed moisture.",
                      f"Combine/store at ≤{p.harvest_safe_moisture_pct:.0f}%. Use a slow cylinder and lifters "
                      "to limit splitting and harvest loss on a lodged crop.", state.day_of_season)]

    def _estimate_harvest_date(self, state: PeaFieldState) -> None:
        p = self.params
        season_days = (p.total_season_days_min + p.total_season_days_max) // 2
        state.estimated_harvest_date = state.seeding_date + timedelta(days=season_days)

    def _build_season_summary(self, state: PeaFieldState, alerts: list[Alert]) -> dict[str, Any]:
        crit = [a for a in alerts if a.severity == _S.CRITICAL]
        warn = [a for a in alerts if a.severity == _S.WARNING]
        info = [a for a in alerts if a.severity == _S.INFO]
        return {
            "field_id": state.field_id, "pea_type": state.pea_type.value,
            "preceding_crop": state.preceding_crop.value,
            "seeding_date": state.seeding_date.isoformat(),
            "estimated_harvest_date": (state.estimated_harvest_date.isoformat()
                                       if state.estimated_harvest_date else None),
            "final_growth_stage": state.growth_stage.name,
            "final_population_per_m2": state.plant_population_per_m2,
            "yield_potential_t_ha": state.yield_potential_t_ha,
            "yield_potential_bu_ac": state.yield_potential_bu_ac,
            "estimated_protein_pct": state.estimated_protein_pct,
            "yield_breakdown": state.yield_breakdown,
            "total_alerts": len(alerts), "critical_count": len(crit),
            "warning_count": len(warn), "info_count": len(info),
            "critical_alerts": [str(a) for a in crit], "warning_alerts": [str(a) for a in warn],
        }


def pea_seeding_rate(target_plants_per_m2: float, thousand_kernel_weight_g: float = 235.0,
                     emergence_pct: float = 88.0) -> dict[str, float]:
    """Yellow-pea seeding rate from target plant stand and TKW (peas: large seed ~150-280 g)."""
    seeds_per_m2 = target_plants_per_m2 / (emergence_pct / 100.0)
    kg_per_ha = seeds_per_m2 * thousand_kernel_weight_g / 1000.0 * 10.0
    return {
        "seeds_per_m2_needed": round(seeds_per_m2, 1),
        "seeding_rate_kg_per_ha": round(kg_per_ha, 1),
        "seeding_rate_bu_per_ac": round(kg_per_ha / PEA_BU_AC_TO_KG_HA, 2),
        "target_plants_per_m2": target_plants_per_m2,
    }
