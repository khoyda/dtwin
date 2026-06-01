# Canola Digital Twin (Canada)

A digital-twin (DT) simulation for **canola (*Brassica napus*) grown on the Canadian Prairies**
(Saskatchewan, Alberta, Manitoba). The twin couples a lightweight **process-based growth model**
(phenology + water balance driven by weather) with a **data-driven yield predictor**
(scikit-learn), so the simulated crop state can be continuously reconciled against observations.

**Primary goals:**

1. **Yield prediction** — end-of-season yield (kg/ha) from in-season weather and management inputs.
2. **Crop timing (phenology)** — when each growth stage is reached, and a forward forecast of
   upcoming stage dates to support field-operation scheduling (e.g. fungicide at early flowering,
   swathing/harvest readiness at maturity).

Canola on the Prairies is overwhelmingly rain-fed, so the simulation emphasizes growing-degree-days
(GDD), water balance, and **heat stress during flowering** rather than irrigation scheduling. Crop
timing is driven by cumulative GDD crossing stage thresholds (see `agronomy.stage_gdd_thresholds`
in `config.yaml`).

## Why a digital twin?

Following the DT framing in **Purcell & Neubauer (2023)** and **Kim & Heo (2024)**, this project treats
the field as a *physical twin* and maintains a *virtual twin* that:

1. **Ingests** weather + soil + management data for a field-season.
2. **Simulates** crop development (phenology, biomass, soil-water) forward in time.
3. **Predicts** yield with an ML model trained on historical field-seasons.
4. **Assimilates** new observations to correct simulated state (planned; see roadmap).

## Project layout

```
canola-digital-twin/
├── config/config.yaml          # paths, model + agronomic settings
├── data/
│   ├── raw/                     # untouched source data (weather, yield, soil)
│   ├── processed/               # cleaned, feature-ready tables
│   └── external/                # reference datasets (e.g. SoilGrids, AAFC)
├── notebooks/                   # exploratory analysis
├── src/canola_dt/
│   ├── config.py                # load/validate config.yaml
│   ├── constants.py             # canola agronomic constants (GDD base, stages…)
│   ├── data/ingest.py           # load weather / yield / soil sources
│   ├── data/preprocess.py       # clean + align to daily field-season frames
│   ├── features.py              # GDD, heat-stress, water-balance features
│   ├── models/yield_model.py    # sklearn yield-prediction pipeline
│   └── simulation/
│       ├── growth.py            # phenology + simple water/biomass dynamics
│       ├── phenology.py         # crop timing: stage timeline + forward forecast
│       └── twin.py              # CanolaDigitalTwin orchestrator
├── scripts/run_simulation.py    # end-to-end demo on synthetic data
└── tests/                       # smoke + unit tests
```

## Getting started

> **Prerequisite:** Python 3.11+ is not yet installed on this machine. Install it from
> https://www.python.org/downloads/windows/ (tick *Add python.exe to PATH*), then:

```powershell
cd C:\Users\kyleh\projects\canola-digital-twin
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Run the end-to-end demo on synthetic weather data
python scripts/run_simulation.py

# Train the yield model on REAL data (downloads + caches ECCC & StatCan on first run)
python scripts/train_yield_model.py

# Run tests
pytest
```

## Real data pipeline

`scripts/train_yield_model.py` builds a training set from live public sources and
fits the configured model:

- **Weather — ECCC** ([`data/eccc.py`](src/canola_dt/data/eccc.py)): daily climate
  data pulled from the Environment & Climate Change Canada bulk endpoint for one
  long-record agricultural (CDA) station per Prairie province, cached per station-year.
  Defaults: Indian Head CDA (SK), Lethbridge CDA (AB), Carman U of M CS (MB). Growing
  season (May 1–Sep 30) features are aggregated via the same twin used at inference.
- **Yield — StatCan** ([`data/statcan.py`](src/canola_dt/data/statcan.py)): canola
  *average yield (kg/ha)* by province by year from Table 32-10-0359, via the Web Data
  Service full-table download.
- **Yield — AAFC/SCIC** ([`data/aafc.py`](src/canola_dt/data/aafc.py)): *optional*
  sub-provincial yields. AAFC's provincial field-crop estimates largely mirror StatCan,
  so the additive value is at the **sub-provincial** scale (SCIC/MASC Rural-Municipality
  yields). There's no single stable API URL for these, so the loader takes a user-supplied
  CSV — set `data_sources.aafc.yield_csv` and see that module's docstring for the schema.

**Current result (1995–2023, 3 provinces, 78 samples):** 5-fold CV R² ≈ **0.59**, MAE ≈
**197 kg/ha** (~11% of the ~1800 kg/ha mean). The dominant feature is `year` — canola
yields carry a strong upward technology/genetics trend that weather can't explain;
weather features (min temp, water stress, precip, heat-stress days) add the remaining
signal. To make *weather* the primary driver, train on sub-provincial yields matched to
local weather (the AAFC/SCIC path above).

## Data sources (suggested)

| Layer        | Source                                                            |
|--------------|-------------------------------------------------------------------|
| Weather      | Environment & Climate Change Canada (ECCC); NASA POWER (gridded)  |
| Yield (hist) | Statistics Canada Table 32-10-0359; AAFC crop reports             |
| Soil         | SoilGrids; AAFC Soil Landscapes of Canada (SLC)                   |
| Phenology    | Canola Council of Canada growth-stage guide (BBCH-aligned)        |

## Roadmap

- [x] Scaffold + synthetic end-to-end pipeline
- [x] Real ECCC weather ingestion (bulk daily, cached per station-year)
- [x] Historical yield join (StatCan) and model training (CV R² ≈ 0.59)
- [ ] Sub-provincial AAFC/SCIC yields matched to local weather (weather-driven model)
- [ ] More stations per province (province-mean weather; capture spatial variation)
- [ ] Data assimilation step (Kalman/EnKF) to correct simulated state
- [ ] Calibration of GDD thresholds & heat-stress response to Prairie data

## Key references

- Purcell, W., & Neubauer, T. (2023). *Digital Twins in Agriculture: A State-of-the-Art Review.*
- Kim, S., & Heo, ... (2024). *[Digital twin application in agriculture].*
