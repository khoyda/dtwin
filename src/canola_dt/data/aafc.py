"""AAFC / crop-insurance sub-provincial canola-yield ingestion.

**Data-source note.** AAFC's *principal field crop* yield estimates are derived from
the same StatCan survey exposed in :mod:`canola_dt.data.statcan`, so they add little
at the provincial scale. The genuinely *additive* sub-provincial sources are the
crop-insurance yield series that AAFC aggregates and republishes:

* **SCIC** (Saskatchewan Crop Insurance Corp.) — yields by Rural Municipality (RM).
* **MASC** (Manitoba Agricultural Services Corp.) — yields by RM / risk area.
* AAFC Census Agricultural Region (CAR) summaries.

These come as plain CSVs but have no single stable API URL, so this module loads a
**user-supplied CSV** (path set via ``config.yaml -> data_sources.aafc.yield_csv``)
rather than fabricating a download. Point it at a SCIC/MASC/AAFC export with the
columns below and it slots straight into the training set.

Expected CSV schema (extra columns ignored)::

    region, year, yield, unit[, province, lat, lon]

where ``unit`` is one of ``kg/ha`` or ``bu/ac`` (canola).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canola: 1 bushel = 50 lb = 22.6796 kg; 1 acre = 0.404686 ha  ->  1 bu/ac = 56.06 kg/ha.
CANOLA_BU_AC_TO_KG_HA = 22.6796 / 0.404686
# Wheat: 1 bushel = 60 lb = 27.2155 kg  ->  1 bu/ac = 67.25 kg/ha.
WHEAT_BU_AC_TO_KG_HA = 27.2155 / 0.404686


def _to_kg_ha(value: float, unit: str) -> float:
    u = str(unit).strip().lower().replace(" ", "")
    if u in {"kg/ha", "kgha", "kilogramsperhectare"}:
        return float(value)
    if u in {"bu/ac", "buac", "bushelsperacre"}:
        return float(value) * CANOLA_BU_AC_TO_KG_HA
    raise ValueError(f"unsupported canola yield unit: {unit!r}")


def load_region_yield(csv_path: str | Path) -> pd.DataFrame:
    """Load a sub-provincial canola-yield CSV into a normalized frame.

    Returns columns ``region, province, year, yield_kg_ha`` (``province`` may be
    NaN if the source omits it). Yields are converted to kg/ha as needed.
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    region = df[cols["region"]]
    year = df[cols["year"]].astype(int)
    unit_col = cols.get("unit")
    units = df[unit_col] if unit_col else "kg/ha"
    raw_yield = df[cols["yield"]]

    if unit_col:
        kg_ha = [_to_kg_ha(v, u) for v, u in zip(raw_yield, units)]
    else:
        kg_ha = [_to_kg_ha(v, "kg/ha") for v in raw_yield]

    out = pd.DataFrame(
        {
            "region": region.astype(str).values,
            "province": df[cols["province"]].values if "province" in cols else pd.NA,
            "year": year.values,
            "yield_kg_ha": kg_ha,
        }
    )
    return out.dropna(subset=["yield_kg_ha"]).reset_index(drop=True)
