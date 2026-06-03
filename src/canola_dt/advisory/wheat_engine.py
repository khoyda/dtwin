"""Spring-wheat advisory engine (application layer).

Advances Zadoks growth stage and emits decision-support alerts (plant population,
FHB fungicide timing, wheat-midge window, leaf-disease T1/T2, aphids, lodging,
N-for-protein, harvest readiness, rotation). Yield comes from the calibrated wheat
process model x management modifiers (population, rotation, N) the model doesn't
represent; protein is estimated from N relative to the yield-maximizing rate.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from canola_dt.advisory.agronomy import AlertSeverity
from canola_dt.advisory.state import Alert
from canola_dt.advisory.wheat_agronomy import (
    LB_PER_AC_TO_KG_PER_HA,
    WheatAgronomyParameters,
    WheatClass,
    WheatGrowthStage,
)
from canola_dt.advisory.wheat_state import WheatFieldState
from canola_dt.data.aafc import WHEAT_BU_AC_TO_KG_HA
from canola_dt.simulation.wheat_model import WheatCropModel, WheatParameters

_S = AlertSeverity


class WheatAdvisoryEngine:
    """Decision-support engine for spring wheat; yield via the calibrated wheat model."""

    def __init__(self, agronomy: WheatAgronomyParameters | None = None,
                 crop_model: WheatCropModel | None = None):
        self.params = agronomy or WheatAgronomyParameters()
        self.crop_model = crop_model or WheatCropModel(WheatParameters())

    @classmethod
    def with_calibrated_model(cls, cfg=None, agronomy: WheatAgronomyParameters | None = None):
        from canola_dt.config import load_config
        cfg = cfg or load_config()
        return cls(agronomy=agronomy, crop_model=WheatCropModel(WheatParameters.from_calibrated(cfg)))

    # ── Public API ───────────────────────────────────────────────────────────

    def step(self, state: WheatFieldState) -> tuple[list[Alert], list[str]]:
        self._update_growth_stage(state)
        alerts: list[Alert] = []
        alerts += self._check_population(state)
        alerts += self._check_rotation(state)
        alerts += self._check_leaf_disease(state)
        alerts += self._check_midge(state)
        alerts += self._check_fhb(state)
        alerts += self._check_aphids(state)
        alerts += self._check_nitrogen_protein(state)
        alerts += self._check_lodging(state)
        alerts += self._check_harvest(state)
        self._estimate_harvest_date(state)
        state.alert_log.extend(alerts)
        return alerts, [a.recommendation for a in alerts if a.recommendation]

    def update_yield(self, state: WheatFieldState, weather, latitude: float | None = None):
        lat = latitude if latitude is not None else state.latitude
        bio = self.crop_model.run(weather, lat).summary["yield_kg_ha"]
        mods = self._management_modifiers(state)
        kg_ha = bio * mods["combined"]
        state.yield_potential_t_ha = round(kg_ha / 1000.0, 2)
        state.yield_potential_bu_ac = round(kg_ha / WHEAT_BU_AC_TO_KG_HA, 1)
        state.estimated_protein_pct = self._estimate_protein(state, kg_ha)
        state.yield_breakdown = {
            "biophysical_kg_ha": round(bio, 1), "population_mod": mods["population"],
            "rotation_mod": mods["rotation"], "nitrogen_mod": mods["nitrogen"],
            "final_kg_ha": round(kg_ha, 1), "estimated_protein_pct": state.estimated_protein_pct,
        }
        return state.yield_breakdown

    def run_season(self, state: WheatFieldState, sensor_readings: list[dict],
                   weather=None, latitude: float | None = None) -> dict[str, Any]:
        all_alerts: list[Alert] = []
        for reading in sensor_readings:
            state.ingest_sensor_reading(**reading)
            alerts, _ = self.step(state)
            all_alerts.extend(alerts)
        if weather is not None:
            self.update_yield(state, weather, latitude)
        return self._build_season_summary(state, all_alerts)

    # ── Yield helpers ─────────────────────────────────────────────────────────

    def _management_modifiers(self, state: WheatFieldState) -> dict[str, float]:
        p = self.params
        pop = state.plant_population_per_m2
        if pop <= 0:
            pop_mod = 1.0
        elif pop >= p.target_population_per_m2_min:
            pop_mod = 1.0
        elif pop >= p.thin_stand_per_m2:
            pop_mod = 0.95
        elif pop >= p.critical_stand_per_m2:
            pop_mod = 0.85
        else:
            pop_mod = 0.70
        rotation_mod = p.preceding_crop_yield_index.get(state.preceding_crop.value, 1.0)
        if state.n_applied_kg_per_ha >= p.N_following_stubble_kg_per_ha_max:
            n_mod = 1.0
        elif state.n_applied_kg_per_ha >= p.N_following_stubble_kg_per_ha_min:
            n_mod = 0.95
        else:
            n_mod = 0.88
        return {"population": round(pop_mod, 3), "rotation": round(rotation_mod, 3),
                "nitrogen": round(n_mod, 3),
                "combined": round(pop_mod * rotation_mod * n_mod, 4)}

    def _estimate_protein(self, state: WheatFieldState, yield_kg_ha: float) -> float:
        """Rough protein (%): rises ~0.3%/10 kg N above the yield-maximizing N rate."""
        p = self.params
        protein = p.protein_target_pct + 0.03 * (state.n_applied_kg_per_ha
                                                 - p.N_min_for_max_yield_CWRS_kg_per_ha)
        return round(max(9.5, min(p.protein_max_with_yield_pct, protein)), 1)

    # ── Phenology (day-count -> Zadoks stage) ─────────────────────────────────

    def _update_growth_stage(self, state: WheatFieldState) -> None:
        dos = state.day_of_season
        G = WheatGrowthStage
        thresholds = [(6, G.GERMINATION), (8, G.EMERGENCE), (22, G.TILLERING),
                      (40, G.JOINTING), (50, G.FLAG_LEAF), (56, G.BOOT),
                      (63, G.HEADING), (70, G.ANTHESIS), (100, G.GRAIN_FILL)]
        state.growth_stage = G.MATURITY
        for day_max, stage in thresholds:
            if dos < day_max:
                state.growth_stage = stage
                break

    # ── Alert checks ──────────────────────────────────────────────────────────

    def _check_population(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        pop = state.plant_population_per_m2
        if pop == 0:
            return []
        if pop >= p.target_population_per_m2_min:
            return []
        if pop >= p.thin_stand_per_m2:
            return [Alert(_S.WARNING, "PlantPopulation",
                          f"Stand {pop:.0f} plants/m² below target "
                          f"({p.target_population_per_m2_min}-{p.target_population_per_m2_max}).",
                          "Slightly thin: expect more tillering and less uniform maturity. Time "
                          "in-crop herbicide early (Z21-29) while canopy competition is reduced.",
                          state.day_of_season)]
        if pop >= p.critical_stand_per_m2:
            return [Alert(_S.WARNING, "PlantPopulation",
                          f"Thin stand {pop:.0f} plants/m² (target ≥{p.target_population_per_m2_min}). "
                          "Excess tillering, uneven maturity and reduced yield stability.",
                          "Prioritise weed control. Expect staggered heading/anthesis — widen FHB and "
                          "midge scouting windows. Consider this in harvest timing.",
                          state.day_of_season)]
        return [Alert(_S.CRITICAL, "PlantPopulation",
                      f"Very thin stand {pop:.0f} plants/m² (<{p.critical_stand_per_m2}). "
                      "Major yield loss and very uneven maturity.",
                      "Evaluate reseed economics; if keeping the stand, do not over-invest in inputs.",
                      state.day_of_season)]

    def _check_rotation(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        if state.day_of_season > 8:
            return []
        from canola_dt.advisory.wheat_agronomy import WheatPrecedingCrop
        if state.preceding_crop in (WheatPrecedingCrop.WHEAT, WheatPrecedingCrop.CEREAL) \
                or state.years_since_last_wheat < p.rotation_min_years:
            sev = _S.CRITICAL if state.preceding_crop == WheatPrecedingCrop.WHEAT else _S.WARNING
            return [Alert(sev, "CropRotation",
                          f"Cereal/wheat preceding crop ({state.preceding_crop.value}), "
                          f"{state.years_since_last_wheat} yr since wheat. Elevated FHB, wheat midge "
                          "and tan spot / septoria risk (residue-borne).",
                          f"Rotate ≥{p.rotation_min_years} yr away from cereals (pulses, canola). Use "
                          "FHB-resistant variety and plan a flowering fungicide; scout for midge.",
                          state.day_of_season)]
        return []

    def _check_leaf_disease(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        rust = max(state.disease_pressure.get("leaf_rust", 0.0),
                   state.disease_pressure.get("stripe_rust", 0.0))
        if state.growth_stage == p.leaf_fungicide_T1_stage and (
                state.leaf_disease_severity_pct > 5 or rust > 0.3):
            return [Alert(_S.WARNING, "LeafDisease",
                          f"Leaf disease at flag leaf (Z39) — severity {state.leaf_disease_severity_pct:.0f}%, "
                          f"rust pressure {rust:.2f}. Flag-leaf (T1) is the key foliar timing.",
                          "Apply a flag-leaf fungicide (rust/septoria/tan spot). Protecting the flag leaf "
                          "gives the most economic foliar-disease response.",
                          state.day_of_season)]
        if state.growth_stage == p.leaf_fungicide_T2_stage and (
                state.leaf_disease_severity_pct > 10 or rust > 0.4):
            return [Alert(_S.WARNING, "LeafDisease",
                          f"Leaf disease persisting at head emergence (Z55-59), severity "
                          f"{state.leaf_disease_severity_pct:.0f}%. T2 window.",
                          "Consider a T2 fungicide for leaf disease, but DO NOT rely on heading-only "
                          "timing for FHB — that needs the anthesis (Z60-65) application.",
                          state.day_of_season)]
        return []

    def _check_midge(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        in_window = p.midge_window_open_stage <= state.growth_stage <= p.midge_window_close_stage
        if not in_window:
            return []
        if state.may_precipitation_mm < p.midge_emergence_precip_min_mm:
            return [Alert(_S.INFO, "WheatMidge",
                          f"In the midge-susceptible window (boot-anthesis) but May rainfall "
                          f"{state.may_precipitation_mm:.0f} mm is below the ~{p.midge_emergence_precip_min_mm:.0f} mm "
                          "needed for strong emergence — risk likely low.",
                          "Scout at sunset to confirm; spray only if the economic threshold is met.",
                          state.day_of_season)]
        if state.midge_per_head >= p.midge_threshold_yield_per_head:
            return [Alert(_S.CRITICAL, "WheatMidge",
                          f"Wheat midge {state.midge_per_head:.2f}/head meets the yield threshold "
                          f"(~1 per {1/p.midge_threshold_yield_per_head:.0f} heads) in the susceptible "
                          "window (boot-anthesis).",
                          "Apply insecticide at sunset during the window. No control once past anthesis. "
                          "Edge infestations are common — a margin spray may suffice.",
                          state.day_of_season)]
        return []

    def _check_fhb(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage != WheatGrowthStage.ANTHESIS:
            return []
        favourable = (state.relative_humidity_pct >= p.fhb_favourable_humidity_pct
                      and p.fhb_favourable_temp_min_c <= state.air_temp_max_c <= p.fhb_favourable_temp_max_c) \
            or state.disease_pressure.get("fusarium_head_blight", 0.0) > 0.4
        if favourable:
            state.fhb_risk_events += 1
        sev = _S.CRITICAL if favourable else _S.WARNING
        return [Alert(sev, "FusariumHeadBlight",
                      f"Anthesis (Z{p.fhb_fungicide_zadoks_start}-{p.fhb_fungicide_zadoks_end}) — FHB "
                      f"fungicide window OPEN. Conditions {'FAVOURABLE' if favourable else 'lower-risk'} "
                      f"(RH {state.relative_humidity_pct:.0f}%, Tmax {state.air_temp_max_c:.0f}°C).",
                      "Apply a triazole (prothioconazole/metconazole) at early-to-mid flower; effective up "
                      f"to ~{p.fhb_max_days_after_anthesis} days after anthesis. Do NOT use a strobilurin "
                      "from boot onward (raises DON). Heading-only timing is insufficient for FHB.",
                      state.day_of_season)]

    def _check_aphids(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        if state.aphids_per_stem <= 0:
            return []
        seedling = state.growth_stage <= WheatGrowthStage.JOINTING
        threshold = p.aphid_threshold_seedling_per_stem if seedling else p.aphid_threshold_boot_per_stem
        if state.growth_stage >= WheatGrowthStage.GRAIN_FILL:
            return []  # do not treat dough->maturity
        if state.aphids_per_stem >= threshold:
            return [Alert(_S.WARNING, "Aphids",
                          f"Aphids {state.aphids_per_stem:.0f}/stem ≥ threshold ({threshold}/stem) "
                          f"at {state.growth_stage.name}.",
                          "Consider control; preserve beneficials where possible. Seedling-stage aphids "
                          "also vector barley yellow dwarf virus.",
                          state.day_of_season)]
        return []

    def _check_nitrogen_protein(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        if state.day_of_season <= 8 and state.n_applied_kg_per_ha < p.N_following_stubble_kg_per_ha_min:
            return [Alert(_S.WARNING, "Nitrogen",
                          f"Applied N {state.n_applied_kg_per_ha:.0f} kg/ha is below the stubble range "
                          f"({p.N_following_stubble_kg_per_ha_min:.0f}-{p.N_following_stubble_kg_per_ha_max:.0f} "
                          "kg/ha). Yield and protein both at risk; protein <13.5% indicates N shortfall.",
                          "Top up N (mid-row band preferred). Split application can lift protein. Note wheat "
                          "does not respond to N in dry years — weigh soil moisture.",
                          state.day_of_season)]
        if state.growth_stage == WheatGrowthStage.BOOT:
            return [Alert(_S.INFO, "ProteinManagement",
                          "Boot stage (Z45): N taken up from here primarily raises PROTEIN, not yield.",
                          f"To lift protein toward {p.protein_target_pct}%+, a post-boot application of "
                          f"~{p.post_boot_N_for_protein_kg_per_ha_min:.0f}-"
                          f"{p.post_boot_N_for_protein_kg_per_ha_max:.0f} kg N/ha can help. "
                          "Avoid strobilurin fungicides from this stage (DON risk).",
                          state.day_of_season)]
        return []

    def _check_lodging(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        if state.lodging_pct <= 0:
            return []
        return [Alert(_S.WARNING, "Lodging",
                      f"Lodging observed ({state.lodging_pct:.0f}% of field). Lodging can cost "
                      f"{p.lodging_yield_loss_pct_min:.0f}-{p.lodging_yield_loss_pct_max:.0f}% yield plus "
                      "harvest losses and disease.",
                      "Adjust harvest plan (slower, one direction). Next year: moderate N, avoid excess "
                      "seeding rate, consider a PGR and stronger-strawed variety.",
                      state.day_of_season)]

    def _check_harvest(self, state: WheatFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage != WheatGrowthStage.MATURITY:
            return []
        return [Alert(_S.INFO, "HarvestReadiness",
                      f"Physiological maturity (Z90s). Grain moisture {state.grain_moisture_pct:.0f}%. "
                      f"Swath at ~{p.swath_kernel_moisture_pct:.0f}%; straight-combine at "
                      f"≤{p.combine_no_dry_moisture_pct:.0f}% (or ≤{p.combine_with_dry_moisture_pct:.0f}% "
                      "with drying).",
                      f"Harvest promptly — delays risk lodging, sprouting (low falling number) and disease. "
                      f"Store ≤{p.storage_safe_moisture_pct:.1f}%. If FHB present, check DON before marketing.",
                      state.day_of_season)]

    def _estimate_harvest_date(self, state: WheatFieldState) -> None:
        p = self.params
        season_days = (p.total_season_days_min + p.total_season_days_max) // 2
        state.estimated_harvest_date = state.seeding_date + timedelta(days=season_days)

    def _build_season_summary(self, state: WheatFieldState, alerts: list[Alert]) -> dict[str, Any]:
        crit = [a for a in alerts if a.severity == _S.CRITICAL]
        warn = [a for a in alerts if a.severity == _S.WARNING]
        info = [a for a in alerts if a.severity == _S.INFO]
        return {
            "field_id": state.field_id, "wheat_class": state.wheat_class.value,
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
            "fhb_risk_events": state.fhb_risk_events,
            "total_alerts": len(alerts), "critical_count": len(crit),
            "warning_count": len(warn), "info_count": len(info),
            "critical_alerts": [str(a) for a in crit], "warning_alerts": [str(a) for a in warn],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone wheat calculators
# ──────────────────────────────────────────────────────────────────────────────

def wheat_seeding_rate(target_plants_per_m2: float, kernel_weight_g: float,
                       survival_pct: float = 85.0) -> dict[str, float]:
    """Wheat seeding rate from a target plant population and thousand-kernel weight.

    Modern recommendation targets ~250-300 plants/m² after ~10-20% stand loss.
    """
    seeds_per_m2 = target_plants_per_m2 / (survival_pct / 100.0)
    kg_per_ha = seeds_per_m2 * kernel_weight_g / 1000.0 * 10.0  # g/m² -> kg/ha
    return {
        "seeds_per_m2_needed": round(seeds_per_m2, 1),
        "seeding_rate_kg_per_ha": round(kg_per_ha, 1),
        "seeding_rate_lb_per_ac": round(kg_per_ha / LB_PER_AC_TO_KG_PER_HA, 1),
        "target_plants_per_m2": target_plants_per_m2,
        "kernel_weight_g": kernel_weight_g,
        "assumed_survival_pct": survival_pct,
    }


def wheat_n_requirement(target_yield_t_ha: float, protein_target_pct: float = 13.5,
                        soil_n_kg_per_ha: float = 40.0,
                        params: WheatAgronomyParameters | None = None) -> dict[str, float]:
    """Estimate N for a target yield + protein (≈30 kg N/t crop demand; protein top-up)."""
    p = params or WheatAgronomyParameters()
    n_per_tonne = 30.0
    yield_demand = target_yield_t_ha * n_per_tonne
    protein_topup = (p.post_boot_N_for_protein_kg_per_ha_max
                     if protein_target_pct >= p.protein_target_pct else 0.0)
    recommended = max(0.0, yield_demand - soil_n_kg_per_ha) + protein_topup
    return {
        "target_yield_t_ha": target_yield_t_ha,
        "protein_target_pct": protein_target_pct,
        "yield_n_demand_kg_per_ha": round(yield_demand, 1),
        "soil_n_kg_per_ha": soil_n_kg_per_ha,
        "protein_topup_kg_per_ha": round(protein_topup, 1),
        "n_recommended_kg_per_ha": round(recommended, 1),
    }
