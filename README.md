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

# Run tests
pytest
```

## Data sources (suggested)

| Layer        | Source                                                            |
|--------------|-------------------------------------------------------------------|
| Weather      | Environment & Climate Change Canada (ECCC); NASA POWER (gridded)  |
| Yield (hist) | Statistics Canada Table 32-10-0359; AAFC crop reports             |
| Soil         | SoilGrids; AAFC Soil Landscapes of Canada (SLC)                   |
| Phenology    | Canola Council of Canada growth-stage guide (BBCH-aligned)        |

## Roadmap

- [x] Scaffold + synthetic end-to-end pipeline
- [ ] Real ECCC / NASA POWER weather ingestion
- [ ] Historical yield join (StatCan / AAFC) and model training
- [ ] Data assimilation step (Kalman/EnKF) to correct simulated state
- [ ] Calibration of GDD thresholds & heat-stress response to Prairie data

## Key references

- Purcell, W., & Neubauer, T. (2023). *Digital Twins in Agriculture: A State-of-the-Art Review.*
- Kim, S., & Heo, ... (2024). *[Digital twin application in agriculture].*
