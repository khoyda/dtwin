"""Advisory engine (application layer) of the agricultural digital twin.

Implements the 'monitor -> evaluate -> recommend' loop: advances growth stage,
evaluates environmental/pest/disease/nutrient/rotation thresholds, generates alerts,
and estimates yield. Yield is **not** a heuristic here — it comes from the calibrated
biophysical process model (:class:`canola_dt.simulation.process_model.CanolaCropModel`),
multiplied by management modifiers (plant density, rotation, N adequacy) that the
biophysical model does not represent. Heat and water stress are left to the process
model to avoid double-counting.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from canola_dt.advisory.agronomy import (
    AgronomyParameters,
    AlertSeverity,
    CultivarType,
    GrowthStage,
    PrecedingCrop,
)
from canola_dt.advisory.state import Alert, CanolaFieldState
from canola_dt import fertility as fert
from canola_dt.data.aafc import CANOLA_BU_AC_TO_KG_HA
from canola_dt.simulation.process_model import CanolaCropModel, CanolaParameters


def _applied_nutrients(state: CanolaFieldState) -> dict[str, float]:
    return {"N": state.n_applied_kg_per_ha, "P2O5": state.p2o5_applied_kg_per_ha,
            "K2O": state.k2o_applied_kg_per_ha, "S": state.s_applied_kg_per_ha}


def _soil_supply(state: CanolaFieldState) -> dict[str, float]:
    return {"N": state.soil_available_n_kg_per_ha, "P2O5": state.soil_available_p2o5_kg_per_ha,
            "K2O": state.soil_available_k2o_kg_per_ha, "S": state.soil_available_s_kg_per_ha}


class CanolaAdvisoryEngine:
    """Decision-support engine; couples agronomic alerts with the process-model yield."""

    def __init__(
        self,
        agronomy: AgronomyParameters | None = None,
        crop_model: CanolaCropModel | None = None,
    ):
        self.params = agronomy or AgronomyParameters()
        # Default to an uncalibrated process model; use with_calibrated_model() for the
        # calibrated one (see scripts/run_advisory.py).
        self.crop_model = crop_model or CanolaCropModel(CanolaParameters())

    @classmethod
    def with_calibrated_model(cls, cfg=None, agronomy: AgronomyParameters | None = None):
        """Build an engine whose yield uses the calibrated process-model parameters."""
        from canola_dt.config import load_config
        cfg = cfg or load_config()
        model = CanolaCropModel(CanolaParameters.from_calibrated(cfg))
        return cls(agronomy=agronomy, crop_model=model)

    # ── Public API ───────────────────────────────────────────────────────────

    def step(self, state: CanolaFieldState) -> tuple[list[Alert], list[str]]:
        """Run one daily advisory step (alerts only; yield is set via update_yield)."""
        new_alerts: list[Alert] = []
        self._update_growth_stage(state)
        new_alerts += self._check_plant_density(state)
        new_alerts += self._check_environmental_stress(state)
        new_alerts += self._check_pest_disease(state)
        new_alerts += self._check_nutrient_inputs(state)
        new_alerts += self._check_rotation(state)
        new_alerts += self._check_harvest_readiness(state)
        self._estimate_harvest_date(state)
        state.alert_log.extend(new_alerts)
        recommendations = [a.recommendation for a in new_alerts if a.recommendation]
        return new_alerts, recommendations

    def update_yield(self, state: CanolaFieldState, weather, latitude: float | None = None):
        """Set yield from the calibrated process model x management modifiers.

        ``weather`` is a daily weather frame (date, tmin_c, tmax_c, tmean_c, precip_mm)
        for the season — the same input the process model uses elsewhere. The biophysical
        yield captures weather/soil (incl. heat and water stress); the advisory modifiers
        add plant density, preceding-crop rotation and N adequacy.
        """
        lat = latitude if latitude is not None else state.latitude
        biophysical_kg_ha = self.crop_model.run(weather, lat).summary["yield_kg_ha"]

        # Liebig nutrient-limited ceiling (N/P/K/S); N then drops from management modifiers.
        nl = fert.nutrient_limited_yield(_applied_nutrients(state), _soil_supply(state),
                                         params=fert.canola_nutrient_parameters())
        nutrient_ceiling = nl.yield_t_ha * 1000.0
        attainable = min(biophysical_kg_ha, nutrient_ceiling)
        limiting = nl.limiting_nutrient if nutrient_ceiling < biophysical_kg_ha else None

        mods = self._management_modifiers(state, include_nitrogen=False)
        kg_ha = attainable * mods["combined"]
        state.yield_potential_t_ha = round(kg_ha / 1000.0, 2)
        state.yield_potential_bu_ac = round(kg_ha / CANOLA_BU_AC_TO_KG_HA, 1)
        state.yield_breakdown = {
            "biophysical_kg_ha": round(biophysical_kg_ha, 1),
            "nutrient_ceiling_kg_ha": round(nutrient_ceiling, 1),
            "limiting_factor": limiting if limiting else "water/weather",
            "density_mod": mods["density"],
            "rotation_mod": mods["rotation"],
            "final_kg_ha": round(kg_ha, 1),
        }
        return state.yield_breakdown

    def fertility_report(self, state: CanolaFieldState, target_yield_t_ha: float) -> dict[str, Any]:
        """Fertilizer recommendation + nutrient-limited yield + deficiency alerts (canola)."""
        cp = fert.canola_nutrient_parameters()
        soil, applied = _soil_supply(state), _applied_nutrients(state)
        rec = fert.fertilizer_recommendation(target_yield_t_ha, soil, cp)
        nl = fert.nutrient_limited_yield(applied, soil, cp)
        demand = fert.crop_demand(target_yield_t_ha, cp)
        alerts = []
        for n in ("N", "S", "P2O5", "K2O"):
            available = soil[n] + applied[n]
            if available < demand[n]:
                note = {"N": "yellowing on older/lower leaves",
                        "S": "canola is highly S-demanding; deficiency yellows the NEWEST leaves",
                        "P2O5": "seed-row safe limit ~22 kg/ha; band the rest",
                        "K2O": "usually only limiting on sandy/organic soils"}[n]
                alerts.append(f"{n}: available {available:.0f} < crop uptake {demand[n]:.0f} kg/ha "
                              f"for {target_yield_t_ha} t/ha — {note}")
        return {
            "target_yield_t_ha": target_yield_t_ha,
            "recommendation_kg_ha": {n: rec[n]["recommended_kg_ha"] for n in rec},
            "nutrient_limited_yield_t_ha": nl.yield_t_ha,
            "limiting_nutrient": nl.limiting_nutrient,
            "supported_yield_t_ha": nl.supported_t_ha,
            "deficiency_alerts": alerts,
        }

    def run_season(
        self,
        state: CanolaFieldState,
        sensor_readings: list[dict],
        weather=None,
        latitude: float | None = None,
    ) -> dict[str, Any]:
        """Replay a season of sensor readings; set yield from ``weather`` if given."""
        all_alerts: list[Alert] = []
        for reading in sensor_readings:
            state.ingest_sensor_reading(**reading)
            alerts, _ = self.step(state)
            all_alerts.extend(alerts)
        if weather is not None:
            self.update_yield(state, weather, latitude)
        return self._build_season_summary(state, all_alerts)

    # ── Yield: management modifiers on the biophysical yield ──────────────────

    def _management_modifiers(self, state: CanolaFieldState,
                              include_nitrogen: bool = True) -> dict[str, float]:
        """Density / rotation / N modifiers the process model does not represent."""
        p = self.params
        d = state.plant_density_per_m2
        if d <= 0:
            density_mod = 1.0  # not yet counted -> no penalty
        elif d >= p.plant_density_optimal_min:
            density_mod = 1.0  # canola compensates by branching across the optimal 50-80 range
        elif d >= p.plant_density_warning_min:
            density_mod = 0.85
        elif d >= p.plant_density_critical_min:
            density_mod = 0.65
        else:
            density_mod = 0.40

        rotation_mod = p.preceding_crop_yield_index.get(state.preceding_crop.value, 1.0)

        n_mod = 1.0
        if include_nitrogen:
            n_threshold = 100 if state.cultivar_type == CultivarType.HYBRID else 60
            n_mod = 1.0 if state.n_applied_kg_per_ha >= n_threshold else 0.88

        return {
            "density": round(density_mod, 3),
            "rotation": round(rotation_mod, 3),
            "nitrogen": round(n_mod, 3),
            "combined": round(density_mod * rotation_mod * n_mod, 4),
        }

    # ── Growth-stage engine (day-count; drives alert timing) ──────────────────

    def _update_growth_stage(self, state: CanolaFieldState) -> None:
        p = self.params
        dos = state.day_of_season
        if dos < p.days_to_emergence_min:
            state.growth_stage = GrowthStage.GS0_GERMINATION
        elif dos < p.days_to_emergence_max + 5:
            state.growth_stage = GrowthStage.GS1_LEAF_DEV
        elif dos < p.days_to_first_flower_min:
            state.growth_stage = GrowthStage.GS3_STEM_ELONG
        elif dos < p.days_to_first_flower_max:
            state.growth_stage = GrowthStage.GS5_FLOWERING
        elif dos < p.days_to_first_flower_max + p.days_flower_to_seed_fill_min:
            state.growth_stage = GrowthStage.GS7_POD_FILL
        elif dos < p.days_to_first_flower_max + p.days_flower_to_seed_fill_max:
            state.growth_stage = GrowthStage.GS8_SEED_FILL
        else:
            state.growth_stage = GrowthStage.GS9_MATURITY

    # ── Plant density checks ──────────────────────────────────────────────────

    def _check_plant_density(self, state: CanolaFieldState) -> list[Alert]:
        p = self.params
        alerts: list[Alert] = []
        density = state.plant_density_per_m2
        if density == 0:
            return alerts

        if density >= p.plant_density_lodging_risk:
            alerts.append(Alert(
                AlertSeverity.WARNING, "PlantDensity",
                f"Stand density {density:.0f} plants/m² exceeds lodging risk threshold "
                f"({p.plant_density_lodging_risk} plants/m²). Thin stems; sclerotinia risk elevated.",
                "Monitor canopy for lodging. Ensure fungicide timing aligns with 10-30% flower open. "
                "Assess harvest strategy (straight-cut likely difficult).",
                state.day_of_season))
        elif density >= p.plant_density_optimal_min:
            pass
        elif density >= p.plant_density_warning_min:
            maturity_delay = min(
                p.low_density_maturity_delay_max_days,
                int((p.plant_density_optimal_min - density) / p.plant_density_optimal_min
                    * p.low_density_maturity_delay_max_days))
            alerts.append(Alert(
                AlertSeverity.WARNING, "PlantDensity",
                f"Stand density {density:.0f} plants/m² is in the warning zone "
                f"({p.plant_density_warning_min}-{p.plant_density_warning_max} plants/m²). "
                f"Estimated maturity delay: up to {maturity_delay} days.",
                "Increase weed scouting frequency. Lower pest action thresholds. For harvest: target "
                "60% seed colour change on main stem before swathing (thin-stand trigger).",
                state.day_of_season))
        elif density >= p.plant_density_critical_min:
            alerts.append(Alert(
                AlertSeverity.CRITICAL, "PlantDensity",
                f"CRITICAL: Stand density {density:.0f} plants/m² is in the high-risk zone "
                f"({p.plant_density_critical_min}-{p.plant_density_critical_max} plants/m²). "
                f"Significant yield loss expected. Maturity delay up to "
                f"{p.low_density_maturity_delay_max_days} days.",
                "Evaluate reseed economics. If keeping stand: significantly lower all pest thresholds; "
                "use 60% colour change for swath trigger; reduce N rate to avoid rank growth.",
                state.day_of_season))
        elif density > 0:
            alerts.append(Alert(
                AlertSeverity.CRITICAL, "PlantDensity",
                f"CRITICAL: Stand density {density:.0f} plants/m² is below minimum viable threshold "
                f"({p.plant_density_critical_min} plants/m²). Crop failure risk HIGH.",
                "Strongly consider reseeding. Consult crop insurance provider. If maintaining stand, "
                "do not invest in further inputs until stand recovers.",
                state.day_of_season))
        return alerts

    # ── Environmental stress checks ───────────────────────────────────────────

    def _check_environmental_stress(self, state: CanolaFieldState) -> list[Alert]:
        p = self.params
        alerts: list[Alert] = []

        if (state.growth_stage == GrowthStage.GS5_FLOWERING
                and state.air_temp_max_c > p.heat_stress_flowering_threshold_c):
            state.heat_stress_events_at_flowering += 1
            alerts.append(Alert(
                AlertSeverity.WARNING, "HeatStress",
                f"Air temp max {state.air_temp_max_c:.1f}°C exceeds heat stress threshold "
                f"({p.heat_stress_flowering_threshold_c}°C) during flowering (GS5). "
                f"Cumulative heat stress events this season: {state.heat_stress_events_at_flowering}.",
                "No direct in-field intervention. Each 3°C increase in July/Aug max (21->24°C) reduces "
                f"yield by ~{p.heat_yield_loss_t_ha_per_3c_increase} t/ha (Nuttall et al. 1992).",
                state.day_of_season))

        if state.waterlogged_days_consecutive >= p.waterlogging_trigger_days:
            alerts.append(Alert(
                AlertSeverity.CRITICAL, "Waterlogging",
                f"Field waterlogged for {state.waterlogged_days_consecutive} consecutive days "
                f"(trigger: {p.waterlogging_trigger_days} days). Yield reduction expected.",
                "Monitor drainage. Waterlogging reduces root O2 and nutrient uptake. Assess for root "
                "disease. Adjust N timing if soil is saturated to minimise denitrification losses.",
                state.day_of_season))

        if (state.growth_stage in (GrowthStage.GS0_GERMINATION, GrowthStage.GS1_LEAF_DEV)
                and state.air_temp_max_c < 2.0):
            alerts.append(Alert(
                AlertSeverity.WARNING, "FrostRisk",
                f"Low temperature ({state.air_temp_max_c:.1f}°C) during vulnerable seedling stage "
                f"({state.growth_stage.value}). Canola growing point is above soil — more susceptible "
                "than cereals.",
                "Scout for stand damage within 3-5 days. Count survivors vs target density. Reseed if "
                "density falls below 30-40 plants/m².",
                state.day_of_season))
        return alerts

    # ── Pest / disease threshold checks ───────────────────────────────────────

    def _check_pest_disease(self, state: CanolaFieldState) -> list[Alert]:
        p = self.params
        alerts: list[Alert] = []

        if (state.growth_stage in (GrowthStage.GS0_GERMINATION, GrowthStage.GS1_LEAF_DEV)
                and state.leaf_defoliation_pct >= p.flea_beetle_action_threshold_pct):
            alerts.append(Alert(
                AlertSeverity.CRITICAL, "FleaBeetle",
                f"Leaf defoliation {state.leaf_defoliation_pct:.0f}% meets/exceeds flea beetle action "
                f"threshold ({p.flea_beetle_action_threshold_pct}%). Damage can escalate from 25% to "
                "50% within a single day.",
                "Apply foliar insecticide immediately if beetles still actively feeding. Tolerability "
                "threshold is 50% leaf loss; act now at 25%. Check seed treatment efficacy.",
                state.day_of_season))

        if state.growth_stage == GrowthStage.GS5_FLOWERING:
            sclerotinia_risk = state.disease_pressure.get("sclerotinia", 0.0)
            if sclerotinia_risk > 0.5 or state.soil_moisture_pct > 40:
                alerts.append(Alert(
                    AlertSeverity.WARNING, "SclerotiniaStemRot",
                    f"Elevated sclerotinia risk at flowering (GS5). Risk index: {sclerotinia_risk:.2f}. "
                    f"Soil moisture: {state.soil_moisture_pct:.1f}%. Fungicide window open "
                    f"(trigger: >={p.sclerotinia_fungicide_trigger_pct_flower}% flower open on main stem).",
                    f"Apply fungicide (FRAC group rotation). Max yield loss in severe wet years: up to "
                    f"{p.sclerotinia_max_yield_loss_pct:.0f}%. Do NOT swath if significant infection "
                    "present and rain forecast — disease progresses rapidly in wet swaths.",
                    state.day_of_season))

        clubroot_risk = state.disease_pressure.get("clubroot", 0.0)
        if clubroot_risk > 0.3:
            alerts.append(Alert(
                AlertSeverity.CRITICAL, "Clubroot",
                f"Clubroot risk index {clubroot_risk:.2f} is elevated. Spreads via soil movement "
                "(equipment, water, footwear).",
                "Enforce strict equipment sanitation at field entry/exit. Switch to a clubroot-resistant "
                f"variety if reseeding. Minimum {p.rotation_recommended_years}-year canola-free rotation "
                "required to reduce spore load.",
                state.day_of_season))
        return alerts

    # ── Nutrient input checks ─────────────────────────────────────────────────

    def _check_nutrient_inputs(self, state: CanolaFieldState) -> list[Alert]:
        p = self.params
        alerts: list[Alert] = []

        if state.n_seed_row_kg_per_ha > p.N_seed_row_risk_threshold_kg_per_ha:
            alerts.append(Alert(
                AlertSeverity.CRITICAL, "SeedRowNitrogen",
                f"Seed-row N rate {state.n_seed_row_kg_per_ha:.0f} kg/ha exceeds safe threshold "
                f"({p.N_seed_row_risk_threshold_kg_per_ha:.0f} kg/ha as urea). Alberta trials show 90% "
                "chance of emergence reduction.",
                "Reduce seed-row N. Band remaining N separately from the seed row. Low seedbed "
                "utilisation (disc openers) is highest risk; increase SBU if possible.",
                state.day_of_season))

        dry_conditions = state.soil_moisture_pct < 20
        p2o5_limit = (p.P_seed_row_safe_max_dry_kg_per_ha if dry_conditions
                      else p.P_seed_row_safe_max_moist_kg_per_ha)
        if state.p2o5_applied_kg_per_ha > p2o5_limit and state.day_of_season <= 7:
            alerts.append(Alert(
                AlertSeverity.WARNING, "SeedRowPhosphorus",
                f"Seed-row P2O5 rate {state.p2o5_applied_kg_per_ha:.0f} kg/ha exceeds safe limit for "
                f"{'dry' if dry_conditions else 'moist'} conditions ({p2o5_limit:.0f} kg/ha). "
                "Seedling injury risk.",
                "Move excess P to a separate band away from the seed row. First 22 kg P2O5/ha "
                "seed-placed is acceptable under most conditions.",
                state.day_of_season))

        if (state.n_applied_kg_per_ha > p.N_inhibits_P_uptake_above_kg_per_ha
                and state.p2o5_applied_kg_per_ha > 0 and state.day_of_season <= 7):
            alerts.append(Alert(
                AlertSeverity.INFO, "NPInteraction",
                f"N rate {state.n_applied_kg_per_ha:.0f} kg/ha in band exceeds "
                f"{p.N_inhibits_P_uptake_above_kg_per_ha:.0f} kg/ha. Concentrated N can cause "
                "ammonia/nitrite toxicity that inhibits early P uptake.",
                "Consider split-banding N and P to reduce toxicity at the seed. Ensure phosphate is "
                "placed where roots can access it before an N-toxic zone develops.",
                state.day_of_season))
        return alerts

    # ── Rotation check ────────────────────────────────────────────────────────

    def _check_rotation(self, state: CanolaFieldState) -> list[Alert]:
        p = self.params
        alerts: list[Alert] = []
        if state.day_of_season > 7:
            return alerts

        if state.years_since_last_canola < p.rotation_min_years:
            alerts.append(Alert(
                AlertSeverity.CRITICAL, "CropRotation",
                f"Canola grown {state.years_since_last_canola} year(s) after previous canola — below "
                f"minimum {p.rotation_min_years}-year rotation. Blackleg pathotype selection and "
                "clubroot buildup risk is HIGH.",
                f"Minimum {p.rotation_min_years} years between canola crops; "
                f"{p.rotation_recommended_years} recommended. Use highest-rated blackleg resistance. "
                "Scout intensively for blackleg and clubroot.",
                state.day_of_season))
        elif state.years_since_last_canola < p.rotation_recommended_years:
            alerts.append(Alert(
                AlertSeverity.WARNING, "CropRotation",
                f"Canola grown {state.years_since_last_canola} year(s) after previous canola. "
                f"Recommended interval is {p.rotation_recommended_years} years.",
                "Select a variety with MR or R blackleg rating. Monitor for clubroot, verticillium "
                "stripe and blackleg throughout the season.",
                state.day_of_season))

        if state.preceding_crop == PrecedingCrop.CANOLA:
            alerts.append(Alert(
                AlertSeverity.CRITICAL, "CropRotation",
                "Preceding crop was canola. Consecutive canola severely elevates blackleg pathotype "
                "and clubroot risk.",
                "Prioritise disease-resistant varieties. Increase scouting frequency. Consider soil "
                "testing for clubroot resting spores.",
                state.day_of_season))
        return alerts

    # ── Harvest readiness check ───────────────────────────────────────────────

    def _check_harvest_readiness(self, state: CanolaFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage != GrowthStage.GS9_MATURITY:
            return []
        thin_stand = state.plant_density_per_m2 < p.plant_density_warning_min
        swath_trigger = (p.swath_colour_change_thin_stand if thin_stand
                         else p.swath_colour_change_pct_min)
        return [Alert(
            AlertSeverity.INFO, "HarvestReadiness",
            f"Crop has reached maturity (GS9). Swath trigger: {swath_trigger:.0f}% seed colour change "
            f"on main stem ({'thin stand' if thin_stand else 'normal stand'} protocol). "
            f"Current seed moisture: {state.current_seed_moisture_pct:.1f}%.",
            "Monitor seed colour change daily. Swath when trigger reached, OR straight-cut if variety "
            "is shatter-resistant, canopy is well-knit and maturity uniform. Do NOT swath if significant "
            "sclerotinia is present and rain is forecast. Early swathing before 30% seed moisture "
            "reduces oil content.",
            state.day_of_season)]

    # ── Harvest date estimation ───────────────────────────────────────────────

    def _estimate_harvest_date(self, state: CanolaFieldState) -> None:
        p = self.params
        base_season_days = p.days_to_first_flower_max + p.days_flower_to_seed_fill_max
        delay = (p.low_density_maturity_delay_max_days
                 if 0 < state.plant_density_per_m2 < p.plant_density_warning_min else 0)
        state.estimated_harvest_date = state.seeding_date + timedelta(days=base_season_days + delay)

    # ── Season summary ────────────────────────────────────────────────────────

    def _build_season_summary(self, state: CanolaFieldState, alerts: list[Alert]) -> dict[str, Any]:
        critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        warnings = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        infos = [a for a in alerts if a.severity == AlertSeverity.INFO]
        return {
            "field_id": state.field_id,
            "species": state.species.value,
            "cultivar_type": state.cultivar_type.value,
            "preceding_crop": state.preceding_crop.value,
            "seeding_date": state.seeding_date.isoformat(),
            "estimated_harvest_date": (
                state.estimated_harvest_date.isoformat() if state.estimated_harvest_date else None),
            "final_growth_stage": state.growth_stage.value,
            "final_plant_density_per_m2": state.plant_density_per_m2,
            "season_precipitation_mm": state.season_precipitation_mm,
            "heat_stress_events_at_flowering": state.heat_stress_events_at_flowering,
            "total_waterlogged_days": state.total_waterlogged_days,
            "yield_potential_t_ha": state.yield_potential_t_ha,
            "yield_potential_bu_ac": state.yield_potential_bu_ac,
            "yield_breakdown": state.yield_breakdown,
            "total_alerts": len(alerts),
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "info_count": len(infos),
            "critical_alerts": [str(a) for a in critical],
            "warning_alerts": [str(a) for a in warnings],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone agronomic calculators (decision support)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_seeding_rate(
    target_density_per_m2: float,
    thousand_seed_weight_g: float,
    emergence_pct: float = 55.0,
) -> dict[str, float]:
    """Canola seeding rate from target density and thousand-seed weight (Canola Council)."""
    seeds_per_m2_needed = target_density_per_m2 / (emergence_pct / 100)
    kg_per_ha = seeds_per_m2_needed * thousand_seed_weight_g / 1000 * 10  # g/m² -> kg/ha
    lb_per_ac = kg_per_ha * 0.892
    return {
        "seeds_per_m2_needed": round(seeds_per_m2_needed, 1),
        "seeding_rate_kg_per_ha": round(kg_per_ha, 2),
        "seeding_rate_lb_per_ac": round(lb_per_ac, 2),
        "thousand_seed_weight_g": thousand_seed_weight_g,
        "target_density_per_m2": target_density_per_m2,
        "assumed_emergence_pct": emergence_pct,
    }


def estimate_n_requirement(
    target_yield_t_ha: float,
    cultivar_type: CultivarType = CultivarType.HYBRID,
    soil_n_available_kg_per_ha: float = 40.0,
    params: AgronomyParameters | None = None,
) -> dict[str, float]:
    """Estimate total N fertiliser requirement (~55 kg N per tonne of seed yield)."""
    p = params or AgronomyParameters()
    n_per_tonne_kg = 55.0  # kg N per tonne of seed yield (total crop demand)
    total_crop_n_demand = target_yield_t_ha * n_per_tonne_kg
    extra_n_hybrid = (
        (p.hybrid_extra_N_kg_per_ha_min + p.hybrid_extra_N_kg_per_ha_max) / 2
        if cultivar_type == CultivarType.HYBRID else 0
    )
    n_fertiliser_required = max(0, total_crop_n_demand - soil_n_available_kg_per_ha)
    return {
        "target_yield_t_ha": target_yield_t_ha,
        "estimated_crop_n_demand_kg_per_ha": round(total_crop_n_demand, 1),
        "soil_n_available_kg_per_ha": soil_n_available_kg_per_ha,
        "extra_n_for_hybrid_kg_per_ha": round(extra_n_hybrid, 1),
        "n_fertiliser_recommended_kg_per_ha": round(n_fertiliser_required + extra_n_hybrid, 1),
    }


def get_harvest_strategy(
    state: CanolaFieldState,
    params: AgronomyParameters | None = None,
) -> dict[str, Any]:
    """Decision support: recommend swath vs straight-cut."""
    p = params or AgronomyParameters()
    thin_stand = state.plant_density_per_m2 < p.plant_density_warning_min
    shatter_resistant = state.cultivar_type == CultivarType.SHATTER_RESISTANT
    swath_trigger = (p.swath_colour_change_thin_stand if thin_stand
                     else p.swath_colour_change_pct_min)
    straight_cut_viable = (
        shatter_resistant and not thin_stand
        and state.disease_pressure.get("sclerotinia", 0.0) < 0.4
    )
    return {
        "recommended_strategy": "straight_cut" if straight_cut_viable else "swath",
        "swath_trigger_colour_change_pct": swath_trigger,
        "straight_cut_viable": straight_cut_viable,
        "reasons": {
            "thin_stand": thin_stand,
            "shatter_resistant_cultivar": shatter_resistant,
            "sclerotinia_pressure": state.disease_pressure.get("sclerotinia", 0.0),
        },
        "caution": (
            "Do NOT swath if sclerotinia infection is significant AND rain is forecast — disease can "
            "destroy up to 33% of yield in the swath."
            if state.disease_pressure.get("sclerotinia", 0.0) > 0.4 else None
        ),
    }
