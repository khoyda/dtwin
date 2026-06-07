"""Canola fertility / macronutrient-requirement model (N, P, K, S).

Models the crop's macronutrient demand and the fertilizer required to meet a target
yield, plus the *nutrient-limited* yield ceiling (Liebig's law of the minimum) used to
couple fertility into the digital twin's yield estimate.

Agronomic basis (Canola Council of Canada / IPNI nutrient uptake & removal). Canola
takes up and removes, per **tonne of seed yield** (≈ 1 lb/bu × 20 kg/t for canola):

    nutrient   uptake (crop)   removal (in seed)   notes
    N          ~70 kg/t        ~36 kg/t            mobile; season-supplied
    P2O5       ~26 kg/t        ~18 kg/t            immobile; large soil reserve
    K2O        ~56 kg/t        ~11 kg/t            most K stays in straw; prairie soils K-rich
    S          ~11 kg/t        ~7  kg/t            canola is S-demanding; deficiency common

**Recommendation strategy** differs by mobility:
  * N, S  — mobile, supplied within the season → recommend by *uptake deficit*
            (feed the crop): ``uptake×yield − soil_supply``.
  * P, K  — immobile with large reserves → recommend by *removal replacement*
            (maintain soil): ``removal×yield − soil_supply``.

These are **starting defaults** — validate against provincial soil-test calibration
(Saskatchewan / Alberta / Manitoba guidelines) before operational use. Soil-test P and K
(ppm) can be converted to an available kg/ha figure with :func:`ppm_to_kg_ha`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Nutrient(str, Enum):
    N = "N"
    P2O5 = "P2O5"
    K2O = "K2O"
    S = "S"


# Per tonne of seed yield (kg / t). See module docstring for sources.
DEFAULT_UPTAKE = {"N": 70.0, "P2O5": 26.0, "K2O": 56.0, "S": 11.0}
DEFAULT_REMOVAL = {"N": 36.0, "P2O5": 18.0, "K2O": 11.0, "S": 7.0}
DEFAULT_STRATEGY = {"N": "uptake", "S": "uptake", "P2O5": "removal", "K2O": "removal"}

NUTRIENTS = ("N", "P2O5", "K2O", "S")


@dataclass
class NutrientParameters:
    """Macronutrient coefficients (kg per tonne of seed/grain) and strategy.

    Defaults are canola; use :func:`wheat_nutrient_parameters` for spring wheat.
    """
    uptake_kg_per_t: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_UPTAKE))
    removal_kg_per_t: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REMOVAL))
    strategy: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_STRATEGY))


# Spring-wheat coefficients (kg per tonne of grain). Grain is ~2.4% N at 13.5% protein,
# so N removal ~22 kg/t; total uptake (grain + straw) is higher. P/K maintained by removal
# (prairie soils are typically K-rich; straw returns most K), N/S fed within season.
WHEAT_UPTAKE = {"N": 29.0, "P2O5": 11.5, "K2O": 20.0, "S": 3.5}
WHEAT_REMOVAL = {"N": 22.0, "P2O5": 10.0, "K2O": 5.5, "S": 1.8}


def canola_nutrient_parameters() -> "NutrientParameters":
    """Canola macronutrient parameters (the defaults)."""
    return NutrientParameters()


def wheat_nutrient_parameters() -> "NutrientParameters":
    """Spring-wheat macronutrient parameters."""
    return NutrientParameters(uptake_kg_per_t=dict(WHEAT_UPTAKE),
                              removal_kg_per_t=dict(WHEAT_REMOVAL),
                              strategy=dict(DEFAULT_STRATEGY))


# Spring-barley coefficients (kg per tonne of grain). Barley grain is lower-protein than
# wheat (malt 11-12.5%), so N removal is a bit lower; straw holds most of the K.
BARLEY_UPTAKE = {"N": 26.0, "P2O5": 11.0, "K2O": 22.0, "S": 4.0}
BARLEY_REMOVAL = {"N": 19.0, "P2O5": 8.0, "K2O": 6.0, "S": 2.0}


def barley_nutrient_parameters() -> "NutrientParameters":
    """Spring-barley macronutrient parameters."""
    return NutrientParameters(uptake_kg_per_t=dict(BARLEY_UPTAKE),
                              removal_kg_per_t=dict(BARLEY_REMOVAL),
                              strategy=dict(DEFAULT_STRATEGY))


def ppm_to_kg_ha(ppm: float, depth_cm: float = 15.0, bulk_density_kg_m3: float = 1300.0) -> float:
    """Convert a soil-test concentration (ppm = mg/kg) to available kg/ha for a layer.

    Default 0–15 cm at bulk density 1300 kg/m³ gives the familiar ~2 kg/ha per ppm.
    """
    return ppm * (depth_cm / 100.0) * bulk_density_kg_m3 * 10000.0 / 1e6


def crop_demand(target_yield_t_ha: float, params: NutrientParameters | None = None) -> dict[str, float]:
    """Total crop uptake (kg/ha) to attain ``target_yield_t_ha``."""
    p = params or NutrientParameters()
    return {n: round(p.uptake_kg_per_t[n] * target_yield_t_ha, 1) for n in NUTRIENTS}


def seed_removal(target_yield_t_ha: float, params: NutrientParameters | None = None) -> dict[str, float]:
    """Nutrient removed in harvested seed (kg/ha) — the maintenance/replacement amount."""
    p = params or NutrientParameters()
    return {n: round(p.removal_kg_per_t[n] * target_yield_t_ha, 1) for n in NUTRIENTS}


def fertilizer_recommendation(
    target_yield_t_ha: float,
    soil_supply: dict[str, float] | None = None,
    params: NutrientParameters | None = None,
) -> dict[str, dict]:
    """Recommended fertilizer (kg/ha) per nutrient for a target yield.

    Uses the per-nutrient strategy (uptake-deficit for N/S, removal-replacement for P/K),
    crediting plant-available ``soil_supply`` (kg/ha). Returns a per-nutrient breakdown.
    """
    p = params or NutrientParameters()
    soil_supply = soil_supply or {}
    out: dict[str, dict] = {}
    for n in NUTRIENTS:
        uptake = p.uptake_kg_per_t[n] * target_yield_t_ha
        removal = p.removal_kg_per_t[n] * target_yield_t_ha
        demand = uptake if p.strategy[n] == "uptake" else removal
        supply = soil_supply.get(n, 0.0)
        out[n] = {
            "strategy": p.strategy[n],
            "crop_uptake_kg_ha": round(uptake, 1),
            "seed_removal_kg_ha": round(removal, 1),
            "soil_supply_kg_ha": round(supply, 1),
            "recommended_kg_ha": round(max(0.0, demand - supply), 1),
        }
    return out


@dataclass
class NutrientLimitedYield:
    yield_t_ha: float
    limiting_nutrient: str
    supported_t_ha: dict[str, float]  # per-nutrient supported yield


def nutrient_limited_yield(
    applied: dict[str, float] | None = None,
    soil_supply: dict[str, float] | None = None,
    params: NutrientParameters | None = None,
) -> NutrientLimitedYield:
    """Liebig nutrient-limited yield ceiling (t/ha) and the limiting nutrient.

    For each nutrient, the supportable yield is ``(soil_supply + applied) / uptake``;
    the ceiling is the minimum across nutrients. ``soil_supply`` should reflect
    plant-available reserves (large for K on most prairie soils) — otherwise that
    nutrient will appear spuriously limiting.
    """
    p = params or NutrientParameters()
    applied = applied or {}
    soil_supply = soil_supply or {}
    supported: dict[str, float] = {}
    for n in NUTRIENTS:
        available = soil_supply.get(n, 0.0) + applied.get(n, 0.0)
        uptake = p.uptake_kg_per_t[n]
        supported[n] = available / uptake if uptake > 0 else float("inf")
    limiting = min(supported, key=supported.get)
    return NutrientLimitedYield(
        yield_t_ha=round(supported[limiting], 2),
        limiting_nutrient=limiting,
        supported_t_ha={n: round(v, 2) for n, v in supported.items()},
    )
