"""Spring-barley advisory engine (application layer).

Advances Zadoks growth stage and emits decision-support alerts (plant population,
net blotch & scald, FHB, lodging/PGR, cutworm, aphids, N-for-malt-protein, harvest,
rotation). Yield comes from the calibrated barley process model, capped by the
nutrient-limited yield (Liebig over N/P/K/S), then scaled by management modifiers.

The barley-distinctive logic is **malt vs feed**: for malt barley, protein must stay
within 11.0-12.5% (high N raises both yield AND protein); the engine estimates protein
and flags malt-grade risk (protein out of band, or any FHB/DON risk).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from canola_dt.advisory.agronomy import AlertSeverity
from canola_dt.advisory.state import Alert
from canola_dt.advisory.barley_agronomy import (
    BarleyAgronomyParameters,
    BarleyGrowthStage,
    BarleyPrecedingCrop,
)
from canola_dt.advisory.barley_state import BarleyFieldState
from canola_dt import fertility as fert
from canola_dt.data.aafc import BARLEY_BU_AC_TO_KG_HA
from canola_dt.simulation.barley_model import BarleyCropModel, BarleyParameters

_S = AlertSeverity


def _applied_nutrients(state: BarleyFieldState) -> dict[str, float]:
    return {"N": state.n_applied_kg_per_ha, "P2O5": state.p2o5_applied_kg_per_ha,
            "K2O": state.k2o_applied_kg_per_ha, "S": state.s_applied_kg_per_ha}


def _soil_supply(state: BarleyFieldState) -> dict[str, float]:
    return {"N": state.soil_available_n_kg_per_ha, "P2O5": state.soil_available_p2o5_kg_per_ha,
            "K2O": state.soil_available_k2o_kg_per_ha, "S": state.soil_available_s_kg_per_ha}


class BarleyAdvisoryEngine:
    """Decision-support engine for spring barley; yield via the calibrated barley model."""

    def __init__(self, agronomy: BarleyAgronomyParameters | None = None,
                 crop_model: BarleyCropModel | None = None):
        self.params = agronomy or BarleyAgronomyParameters()
        self.crop_model = crop_model or BarleyCropModel(BarleyParameters())

    @classmethod
    def with_calibrated_model(cls, cfg=None, agronomy: BarleyAgronomyParameters | None = None):
        from canola_dt.config import load_config
        cfg = cfg or load_config()
        return cls(agronomy=agronomy, crop_model=BarleyCropModel(BarleyParameters.from_calibrated(cfg)))

    # ── Public API ───────────────────────────────────────────────────────────

    def step(self, state: BarleyFieldState) -> tuple[list[Alert], list[str]]:
        self._update_growth_stage(state)
        alerts: list[Alert] = []
        alerts += self._check_population(state)
        alerts += self._check_rotation(state)
        alerts += self._check_scald(state)
        alerts += self._check_net_blotch(state)
        alerts += self._check_fhb(state)
        alerts += self._check_lodging_pgr(state)
        alerts += self._check_cutworm(state)
        alerts += self._check_aphids(state)
        alerts += self._check_nitrogen_protein(state)
        alerts += self._check_harvest(state)
        self._estimate_harvest_date(state)
        state.alert_log.extend(alerts)
        return alerts, [a.recommendation for a in alerts if a.recommendation]

    def update_yield(self, state: BarleyFieldState, weather, latitude: float | None = None):
        lat = latitude if latitude is not None else state.latitude
        bio = self.crop_model.run(weather, lat).summary["yield_kg_ha"]
        nl = fert.nutrient_limited_yield(_applied_nutrients(state), _soil_supply(state),
                                         params=fert.barley_nutrient_parameters())
        nutrient_ceiling = nl.yield_t_ha * 1000.0
        attainable = min(bio, nutrient_ceiling)
        limiting = nl.limiting_nutrient if nutrient_ceiling < bio else None
        mods = self._management_modifiers(state, include_nitrogen=False)
        kg_ha = attainable * mods["combined"]
        state.yield_potential_t_ha = round(kg_ha / 1000.0, 2)
        state.yield_potential_bu_ac = round(kg_ha / BARLEY_BU_AC_TO_KG_HA, 1)
        state.estimated_protein_pct = self._estimate_protein(state)
        state.malt_grade_ok = self._malt_grade_ok(state)
        state.yield_breakdown = {
            "biophysical_kg_ha": round(bio, 1),
            "nutrient_ceiling_kg_ha": round(nutrient_ceiling, 1),
            "limiting_factor": limiting if limiting else "water/weather",
            "population_mod": mods["population"], "rotation_mod": mods["rotation"],
            "final_kg_ha": round(kg_ha, 1), "estimated_protein_pct": state.estimated_protein_pct,
            "malt_grade_ok": state.malt_grade_ok,
        }
        return state.yield_breakdown

    def run_season(self, state: BarleyFieldState, sensor_readings: list[dict],
                   weather=None, latitude: float | None = None) -> dict[str, Any]:
        all_alerts: list[Alert] = []
        for reading in sensor_readings:
            state.ingest_sensor_reading(**reading)
            alerts, _ = self.step(state)
            all_alerts.extend(alerts)
        if weather is not None:
            self.update_yield(state, weather, latitude)
        return self._build_season_summary(state, all_alerts)

    def fertility_report(self, state: BarleyFieldState, target_yield_t_ha: float) -> dict[str, Any]:
        bp = fert.barley_nutrient_parameters()
        soil, applied = _soil_supply(state), _applied_nutrients(state)
        rec = fert.fertilizer_recommendation(target_yield_t_ha, soil, bp)
        nl = fert.nutrient_limited_yield(applied, soil, bp)
        demand = fert.crop_demand(target_yield_t_ha, bp)
        alerts = []
        for n in ("N", "S", "P2O5", "K2O"):
            if soil[n] + applied[n] < demand[n]:
                alerts.append(f"{n}: available {soil[n] + applied[n]:.0f} < crop uptake "
                              f"{demand[n]:.0f} kg/ha for {target_yield_t_ha} t/ha")
        return {"target_yield_t_ha": target_yield_t_ha,
                "recommendation_kg_ha": {n: rec[n]["recommended_kg_ha"] for n in rec},
                "nutrient_limited_yield_t_ha": nl.yield_t_ha,
                "limiting_nutrient": nl.limiting_nutrient, "deficiency_alerts": alerts}

    # ── Yield / protein helpers ───────────────────────────────────────────────

    def _management_modifiers(self, state: BarleyFieldState,
                              include_nitrogen: bool = True) -> dict[str, float]:
        p = self.params
        pop = state.plant_population_per_m2
        target_min = (p.malt_target_population_per_m2_min if p.is_malt()
                      else p.feed_target_population_per_m2_min)
        if pop <= 0 or pop >= target_min:
            pop_mod = 1.0
        elif pop >= p.thin_stand_per_m2:
            pop_mod = 0.95
        elif pop >= p.critical_stand_per_m2:
            pop_mod = 0.85
        else:
            pop_mod = 0.70
        rotation_mod = p.preceding_crop_yield_index.get(state.preceding_crop.value, 1.0)
        n_mod = 1.0
        if include_nitrogen:
            if state.n_applied_kg_per_ha >= p.N_following_stubble_kg_per_ha_max:
                n_mod = 1.0
            elif state.n_applied_kg_per_ha >= p.N_following_stubble_kg_per_ha_min:
                n_mod = 0.95
            else:
                n_mod = 0.88
        return {"population": round(pop_mod, 3), "rotation": round(rotation_mod, 3),
                "nitrogen": round(n_mod, 3),
                "combined": round(pop_mod * rotation_mod * n_mod, 4)}

    def _estimate_protein(self, state: BarleyFieldState) -> float:
        """Grain protein (%): ~12.5% at the malt-safe N rate, +0.03%/kg N above it."""
        p = self.params
        protein = p.malt_protein_max_pct + 0.03 * (state.n_applied_kg_per_ha - p.N_malt_max_safe_kg_per_ha)
        return round(max(8.5, min(18.0, protein)), 1)

    def _malt_grade_ok(self, state: BarleyFieldState) -> bool:
        if not self.params.is_malt():
            return True  # feed has no protein ceiling
        pr = state.estimated_protein_pct
        return (self.params.malt_protein_min_pct <= pr <= self.params.malt_protein_max_pct
                and state.fhb_risk_events == 0)

    # ── Phenology (day-count -> Zadoks stage) ─────────────────────────────────

    def _update_growth_stage(self, state: BarleyFieldState) -> None:
        dos = state.day_of_season
        G = BarleyGrowthStage
        # Barley is faster than wheat: emergence ~7d, heading ~50-70d, maturity ~80-105d.
        thresholds = [(6, G.GERMINATION), (8, G.EMERGENCE), (20, G.TILLERING),
                      (36, G.JOINTING), (46, G.FLAG_LEAF), (52, G.BOOT),
                      (58, G.HEADING), (64, G.ANTHESIS), (92, G.GRAIN_FILL)]
        state.growth_stage = G.MATURITY
        for day_max, stage in thresholds:
            if dos < day_max:
                state.growth_stage = stage
                break

    # ── Alert checks ──────────────────────────────────────────────────────────

    def _check_population(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        pop = state.plant_population_per_m2
        if pop == 0:
            return []
        target_min = (p.malt_target_population_per_m2_min if p.is_malt()
                      else p.feed_target_population_per_m2_min)
        if pop >= target_min:
            return []
        sev = _S.WARNING if pop >= p.critical_stand_per_m2 else _S.CRITICAL
        extra = ("" if not p.is_malt() else
                 " For malt, thin stands tiller more and can give uneven maturity and variable kernel "
                 "plumpness — watch quality.")
        return [Alert(sev, "PlantPopulation",
                      f"Stand {pop:.0f} plants/m² below target (≥{target_min}).{extra}",
                      "Time herbicide early while canopy competition is reduced; expect more tillering "
                      "and less uniform maturity.", state.day_of_season)]

    def _check_rotation(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.day_of_season > 8:
            return []
        if state.preceding_crop in (BarleyPrecedingCrop.BARLEY, BarleyPrecedingCrop.CEREAL) \
                or state.years_since_last_barley < p.rotation_min_years:
            sev = _S.CRITICAL if state.preceding_crop == BarleyPrecedingCrop.BARLEY else _S.WARNING
            return [Alert(sev, "CropRotation",
                          f"Cereal/barley preceding crop ({state.preceding_crop.value}). Builds net "
                          "blotch, scald, spot blotch and common root rot (residue-borne).",
                          f"Rotate ≥{p.rotation_min_years} yr away from cereals (canola, pulses). Use "
                          "resistant varieties and a seed treatment; plan flag-leaf + FHB fungicides.",
                          state.day_of_season)]
        return []

    def _check_scald(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage == BarleyGrowthStage.JOINTING \
                and (state.scald_severity_pct >= p.scald_economic_threshold_pct
                     or state.disease_pressure.get("scald", 0.0) > 0.3):
            return [Alert(_S.WARNING, "Scald",
                          f"Scald {state.scald_severity_pct:.0f}% at jointing (GS31-33) — first-node-to-boot "
                          "is the critical window; cool, wet conditions favour it.",
                          "Apply a fungicide at GS31-45 if pressure is building; use resistant varieties "
                          "and rotate away from cereals next year.", state.day_of_season)]
        return []

    def _check_net_blotch(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage in (BarleyGrowthStage.FLAG_LEAF, BarleyGrowthStage.HEADING) \
                and (state.net_blotch_severity_pct >= p.net_blotch_severity_threshold_pct
                     or state.disease_pressure.get("net_blotch", 0.0) > 0.3):
            note = (" Excess N increases net blotch severity." if p.excess_N_net_blotch_note
                    and state.n_applied_kg_per_ha > p.N_following_stubble_kg_per_ha_max else "")
            return [Alert(_S.WARNING, "NetBlotch",
                          f"Net blotch on the upper canopy at {state.growth_stage.name} "
                          f"(severity {state.net_blotch_severity_pct:.0f}%).{note} Flag leaf/flag-1/flag-2 "
                          "supply most of the grain fill.",
                          "Apply a flag-leaf (Z39) fungicide — the most cost-effective timing. Protect the "
                          "upper three leaves.", state.day_of_season)]
        return []

    def _check_fhb(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage != BarleyGrowthStage.ANTHESIS:
            return []
        favourable = (state.relative_humidity_pct >= p.fhb_favourable_humidity_pct
                      and p.fhb_favourable_temp_min_c <= state.air_temp_max_c <= p.fhb_favourable_temp_max_c) \
            or state.disease_pressure.get("fusarium_head_blight", 0.0) > 0.4
        if favourable:
            state.fhb_risk_events += 1
        sev = _S.CRITICAL if favourable else _S.WARNING
        malt_note = (" For MALT, any detectable DON = rejection (downgrade to feed)."
                     if p.is_malt() else "")
        return [Alert(sev, "FusariumHeadBlight",
                      f"Anthesis (Z{p.fhb_fungicide_zadoks_start}-{p.fhb_fungicide_zadoks_end}) — FHB "
                      f"window OPEN, conditions {'FAVOURABLE' if favourable else 'lower-risk'} "
                      f"(RH {state.relative_humidity_pct:.0f}%, Tmax {state.air_temp_max_c:.0f}°C).{malt_note}",
                      "Apply a triazole at early flower; do NOT use a strobilurin from boot onward "
                      "(raises DON).", state.day_of_season)]

    def _check_lodging_pgr(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.lodging_pct > 0:
            return [Alert(_S.WARNING, "Lodging",
                          f"Lodging {state.lodging_pct:.0f}% — can cost {p.lodging_yield_loss_pct_min:.0f}-"
                          f"{p.lodging_yield_loss_pct_max:.0f}% yield and (for malt) hurt germination and "
                          "plump kernels.",
                          "Harvest carefully. Next year: moderate N/seeding rate, stronger-strawed variety, "
                          "and consider Moddus (trinexapac) at GS30-33 in high-yield fields.",
                          state.day_of_season)]
        if (state.growth_stage == BarleyGrowthStage.JOINTING and not state.pgr_applied
                and state.n_applied_kg_per_ha >= p.N_following_stubble_kg_per_ha_max):
            return [Alert(_S.INFO, "PGR",
                          f"Jointing (Z{p.pgr_moddus_zadoks_min}-{p.pgr_moddus_zadoks_max}) — PGR window. "
                          "High N raises lodging risk.",
                          "In a high-yield (>80 bu/ac) lodging-prone field a Moddus application at GS30-33 "
                          "can cut lodging (no yield effect). Avoid PGRs in low-yield/high-stress seasons; "
                          "for malt, weigh the kernel-weight trade-off.", state.day_of_season)]
        return []

    def _check_cutworm(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage in (BarleyGrowthStage.GERMINATION, BarleyGrowthStage.EMERGENCE,
                                  BarleyGrowthStage.TILLERING) \
                and state.cutworm_larvae_per_m2 >= p.cutworm_pale_western_per_m2:
            return [Alert(_S.WARNING, "Cutworm",
                          f"Cutworm {state.cutworm_larvae_per_m2:.0f}/m² at the seedling stage meets the "
                          f"threshold (~{p.cutworm_pale_western_per_m2:.0f}/m²).",
                          "Scout for bare patches/clipped plants; spot-spray hotspots at dusk.",
                          state.day_of_season)]
        return []

    def _check_aphids(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.aphids_per_stem >= p.aphid_threshold_seedling_per_stem \
                and state.growth_stage <= BarleyGrowthStage.HEADING:
            return [Alert(_S.WARNING, "Aphids",
                          f"Aphids {state.aphids_per_stem:.0f}/stem ≥ threshold "
                          f"({p.aphid_threshold_seedling_per_stem}/stem).",
                          "Consider control; aphids also vector barley yellow dwarf virus, especially at "
                          "the seedling stage.", state.day_of_season)]
        return []

    def _check_nitrogen_protein(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.day_of_season > 8 or not p.is_malt():
            return []
        if state.n_applied_kg_per_ha > p.N_malt_max_safe_kg_per_ha:
            return [Alert(_S.WARNING, "MaltProtein",
                          f"N {state.n_applied_kg_per_ha:.0f} kg/ha exceeds the malt-safe rate "
                          f"(~{p.N_malt_max_safe_kg_per_ha:.0f} kg/ha). Risk of protein >"
                          f"{p.malt_protein_max_pct}% → malt rejection (downgrade to feed).",
                          "For malt, cap N to protect protein; modern varieties (e.g. AAC Synergy) hold "
                          "below 12.5% better than older ones. Excess N also worsens net blotch and lodging.",
                          state.day_of_season)]
        return []

    def _check_harvest(self, state: BarleyFieldState) -> list[Alert]:
        p = self.params
        if state.growth_stage != BarleyGrowthStage.MATURITY:
            return []
        if p.is_malt():
            rec = (f"Swath at ~{p.swath_kernel_moisture_pct:.0f}%. Do NOT combine below "
                   f"{p.malt_combine_floor_moisture_pct:.1f}% — it peels kernels and loses malt grade. "
                   f"Store ≤{p.malt_storage_moisture_max_pct:.1f}%; threshing >16% risks germination loss. "
                   "Use slow cylinder speed to avoid cracking.")
        else:
            rec = (f"Swath at ~{p.swath_kernel_moisture_pct:.0f}%; straight-combine from "
                   f"~{p.combine_straight_cut_start_pct:.0f}%. Store ≤{p.feed_storage_moisture_max_pct:.1f}%.")
        return [Alert(_S.INFO, "HarvestReadiness",
                      f"Maturity (Z90s). Grain moisture {state.grain_moisture_pct:.0f}%. "
                      f"{'Malt' if p.is_malt() else 'Feed'} barley.", rec, state.day_of_season)]

    def _estimate_harvest_date(self, state: BarleyFieldState) -> None:
        p = self.params
        season_days = (p.total_season_days_min + p.total_season_days_max) // 2
        state.estimated_harvest_date = state.seeding_date + timedelta(days=season_days)

    def _build_season_summary(self, state: BarleyFieldState, alerts: list[Alert]) -> dict[str, Any]:
        crit = [a for a in alerts if a.severity == _S.CRITICAL]
        warn = [a for a in alerts if a.severity == _S.WARNING]
        info = [a for a in alerts if a.severity == _S.INFO]
        return {
            "field_id": state.field_id, "barley_type": state.barley_type.value,
            "preceding_crop": state.preceding_crop.value,
            "seeding_date": state.seeding_date.isoformat(),
            "estimated_harvest_date": (state.estimated_harvest_date.isoformat()
                                       if state.estimated_harvest_date else None),
            "final_growth_stage": state.growth_stage.name,
            "final_population_per_m2": state.plant_population_per_m2,
            "yield_potential_t_ha": state.yield_potential_t_ha,
            "yield_potential_bu_ac": state.yield_potential_bu_ac,
            "estimated_protein_pct": state.estimated_protein_pct,
            "malt_grade_ok": state.malt_grade_ok,
            "yield_breakdown": state.yield_breakdown,
            "fhb_risk_events": state.fhb_risk_events,
            "total_alerts": len(alerts), "critical_count": len(crit),
            "warning_count": len(warn), "info_count": len(info),
            "critical_alerts": [str(a) for a in crit], "warning_alerts": [str(a) for a in warn],
        }


def barley_seeding_rate(target_plants_per_m2: float, thousand_kernel_weight_g: float = 45.0,
                        survival_pct: float = 90.0) -> dict[str, float]:
    """Barley seeding rate from target plant population and TKW (~45 g)."""
    seeds_per_m2 = target_plants_per_m2 / (survival_pct / 100.0)
    kg_per_ha = seeds_per_m2 * thousand_kernel_weight_g / 1000.0 * 10.0
    return {
        "seeds_per_m2_needed": round(seeds_per_m2, 1),
        "seeding_rate_kg_per_ha": round(kg_per_ha, 1),
        "seeding_rate_bu_per_ac": round(kg_per_ha / BARLEY_BU_AC_TO_KG_HA, 2),
        "target_plants_per_m2": target_plants_per_m2,
    }
