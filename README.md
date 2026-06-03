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
│   ├── advisory/                # decision-support layer (agronomic alerts + yield)
│   │   ├── agronomy.py          #   Canola Council thresholds + enums (AgronomyParameters)
│   │   ├── state.py             #   CanolaFieldState virtual entity (JSON-serialisable)
│   │   └── engine.py            #   CanolaAdvisoryEngine: alerts + process-model yield
│   └── simulation/
│       ├── growth.py            # lightweight phenology + bucket water balance
│       ├── phenology.py         # crop timing: stage timeline + forward forecast
│       ├── agromet.py           # FAO-56 radiation, daylength, ET0, thermal time
│       ├── process_model.py     # APSIM-style mechanistic canola model
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

## Crop models: two tiers

The project has **two** process models, by design:

- **Lightweight** ([`growth.py`](src/canola_dt/simulation/growth.py)) — GDD phenology +
  single-bucket water balance. Fast, few parameters; drives the twin's interpretable
  state trajectory and the crop-timing forecasts, and is the seam for data assimilation.
- **APSIM-style mechanistic** ([`process_model.py`](src/canola_dt/simulation/process_model.py))
  — a daily-step model with the core processes APSIM/DSSAT represent:
  - cardinal-temperature **thermal time** with a long-day **photoperiod** modifier on the
    pre-floral phase;
  - **canopy** expansion (thermal-time driven) with Beer's-law light interception and
    post-flowering senescence;
  - **biomass** via **radiation-use efficiency** on intercepted PAR, down-regulated by water stress;
  - a **layered cascading soil-water balance** (SCS-CN runoff, tipping-bucket drainage,
    two-source evaporation/transpiration, root-front growth, per-layer uptake);
  - **yield** = maturity biomass × a harvest index reduced by flowering heat and water stress.

  Radiation/ET inputs are derived from temperature + latitude via FAO-56 equations in
  [`agromet.py`](src/canola_dt/simulation/agromet.py), since ECCC daily station data has
  no solar/wind/humidity.

  ```powershell
  python scripts/run_process_model.py 2020   # run on real ECCC weather for a year
  ```

  **Validation status.** Behaviour is correct — a controlled favourable vs. hot-dry season
  gives ~1600 vs. ~400 kg/ha; heat during flowering collapses harvest index; drought raises
  water stress and cuts biomass (see `tests/test_process_model.py`). Absolute yields on real
  SK weather (~1300–1700 kg/ha) read a little low vs. provincial stats — expected, since a
  point station ≠ a province, and **parameters are uncalibrated conventional defaults**.
  Calibrating to Prairie data is described next.

### Calibration

[`calibration.py`](src/canola_dt/calibration.py) + `scripts/calibrate_process_model.py`
calibrate the model against StatCan yields. Two ideas make this honest:

1. **Separate the technology trend from weather.** Observed yields rise ~35–42 kg/ha/yr
   (genetics/agronomy) — something a fixed-genetics process model neither can nor should
   reproduce. We fit a per-province linear trend and express every year's yield at a common
   reference year, so the model is calibrated to the **weather-driven** variation only.
2. **Calibrate each parameter to what it can actually identify** (three steps):
   - **pattern** — grid `kl` × `hi_heat_sensitivity` to maximize the interannual
     **anomaly correlation** (does the model pick the right good/bad years?);
   - **level** — set `rue` so the mean simulated yield matches the mean observed;
   - **offsets** — per-province residuals (a point station ≠ a province).

   ```powershell
   python scripts/calibrate_process_model.py   # writes artifacts/calibrated_params.json
   ```
   The result is loaded automatically wherever `CanolaParameters.from_calibrated(cfg)` is used.

**What calibration found.** Calibration fixes the *level* (mean bias → ~0) and keeps
heat/drought stress physically active. Averaging **five stations per province** (rather than
one) substantially improves things, confirming the spatial-aggregation hypothesis:

| metric                       | 1 station | 5 stations/province |
|------------------------------|-----------|---------------------|
| interannual anomaly corr.    | 0.27      | **0.39**            |
| point-vs-province volatility | 3.5×      | **2.16×**           |
| calibrated RUE (g/MJ)        | 2.11      | **1.57** (realistic)|
| abs. error after offsets     | 1298      | **565 kg/ha**       |

The inflated RUE under one station was compensating for single-point bias; with multiple
stations it settles into the agronomic 1.2–1.7 range. Going further to **ten stations per
province** drives the per-province offsets to ≈0 (level now essentially unbiased), but
interannual **skill plateaus** — anomaly correlation stays ~0.39 and the volatility ratio
does not fall below ~2 (the added dry-region stations are individually more variable). This
is the **provincial-scale ceiling**: more stations fix the *level* but not the *pattern*.
Breaking through needs validation at a scale where local weather maps to local yield —
i.e. **sub-provincial (SCIC/MASC RM-level)** yields.

## Real data pipeline

`scripts/train_yield_model.py` builds a training set from live public sources and
fits the configured model:

- **Weather — ECCC** ([`data/eccc.py`](src/canola_dt/data/eccc.py)): daily climate
  data pulled from the Environment & Climate Change Canada bulk endpoint for **ten
  long-record stations per Prairie province** (30 total), selected by
  `scripts/discover_stations.py` (spread across each province's canola belt, verified for
  completeness) and written to `config/stations.generated.yaml`, cached per station-year.
  Growing-season (May 1–Sep 30) features are aggregated to a province-year mean (the
  spatial average a provincial yield represents) via the same twin used at inference.

  ```powershell
  python scripts/discover_stations.py --per-province 10   # (re)select the station set
  ```
- **Yield — StatCan** ([`data/statcan.py`](src/canola_dt/data/statcan.py)): canola
  *average yield (kg/ha)* by province by year from Table 32-10-0359, via the Web Data
  Service full-table download.
- **Yield — AAFC/SCIC** ([`data/aafc.py`](src/canola_dt/data/aafc.py)): *optional*
  sub-provincial yields. AAFC's provincial field-crop estimates largely mirror StatCan,
  so the additive value is at the **sub-provincial** scale (SCIC/MASC Rural-Municipality
  yields). There's no single stable API URL for these, so the loader takes a user-supplied
  CSV — set `data_sources.aafc.yield_csv` and see that module's docstring for the schema.

**Current result (1995–2023, 3 provinces, 87 samples):** 5-fold CV R² ≈ **0.62**, MAE ≈
**197 kg/ha** (~11% of the ~1830 kg/ha mean). The dominant feature is `year` — canola
yields carry a strong upward technology/genetics trend that weather can't explain;
weather features (water stress, dry days, heat-stress days, precip) add the remaining
signal. To make *weather* the primary driver, train on sub-provincial yields matched to
local weather (the AAFC/SCIC path above).

### Coupling the process model into the ML model

The calibrated APSIM-style model's outputs (simulated yield, biomass, max LAI, harvest
index, phenology timing, water stress, water fluxes) are averaged per province-year and
added as `pm_*` features ([`training.py`](src/canola_dt/training.py) `build_process_features`),
unifying the twin's mechanistic and statistical halves. Feature-layer ablation (5-fold CV R²):

| feature set                | CV R² |
|----------------------------|-------|
| `year` only (trend)        | 0.535 |
| `year` + weather features  | 0.611 |
| `year` + **process only**  | **0.603** |
| `year` + weather + process | **0.625** |

The process-model outputs **alone nearly match the hand-crafted weather features** (0.603 vs
0.611) — the mechanistic model is a compact, physically-grounded summary of the weather→yield
relationship — and combining both is best. The simulated yield's own detrended correlation with
observed yield (≈0.40) is consistent with the calibration. At provincial scale the gains are
modest because the technology trend dominates; the coupling should matter more at the
sub-provincial scale where weather drives a larger share of the variance.

## Data sources (suggested)

| Layer        | Source                                                            |
|--------------|-------------------------------------------------------------------|
| Weather      | Environment & Climate Change Canada (ECCC); NASA POWER (gridded)  |
| Yield (hist) | Statistics Canada Table 32-10-0359; AAFC crop reports             |
| Soil         | SoilGrids; AAFC Soil Landscapes of Canada (SLC)                   |
| Phenology    | Canola Council of Canada growth-stage guide (BBCH-aligned)        |

## Advisory layer (decision support)

[`advisory/`](src/canola_dt/advisory) is the application-layer decision-support front end,
built on Canola Council of Canada agronomic thresholds. `CanolaFieldState` is a
JSON-serialisable virtual entity updated by perception-layer sensor readings;
`CanolaAdvisoryEngine` advances growth stage and emits **alerts** (plant density,
heat/frost/waterlogging, flea beetle, sclerotinia, clubroot, seed-row N/P, rotation,
harvest readiness) plus seeding-rate and N-requirement calculators and a swath vs
straight-cut recommender.

Crucially, **yield is not a heuristic here** — it comes from the calibrated biophysical
process model, multiplied by the management factors the process model doesn't represent
(plant density, preceding-crop rotation, N adequacy); heat and water stress are left to the
process model to avoid double-counting. This unites the mechanistic core with the agronomic
advisory front end:

```powershell
python scripts/run_advisory.py   # alerts over a sensor season + calibrated yield on real weather
```

```
yield = process_model(weather, soil) x density_mod x rotation_mod x N_mod
        └─ calibrated biophysical ─┘   └──── advisory management modifiers ────┘
```

> Note: `advisory/agronomy.py` defines `AgronomyParameters` (threshold constants) — distinct
> from the biophysical `CanolaParameters` in `simulation/process_model.py`.

## Spring wheat (second crop)

A full second-crop digital twin, reusing the crop-agnostic infrastructure (ECCC weather,
FAO-56 agro-met, `calibration.load_season_frames`, the detrend/anomaly calibration helpers)
with wheat-specific biophysics:

- **[`simulation/wheat_model.py`](src/canola_dt/simulation/wheat_model.py)** — `WheatCropModel`:
  cardinal temps base 0/21/35 °C (spring wheat, no vernalization), Zadoks-aligned stages
  (emergence → tillering → jointing → heading → anthesis → grain fill → maturity), RUE biomass,
  layered soil water, harvest index ~0.42 reduced by **grain-fill** heat and water stress.
- **[`wheat_calibration.py`](src/canola_dt/wheat_calibration.py)** — same three-step calibration
  (pattern → level → offsets) against StatCan **"Wheat, spring"** yields.

```powershell
python scripts/run_wheat_model.py 2020        # calibrated wheat yield on real ECCC weather
python scripts/calibrate_wheat_model.py       # calibrate vs StatCan spring-wheat yields
```

**Calibration result (1995–2023, 3 provinces, 87 province-years).** Interannual anomaly
correlation **0.50 → 0.53**, volatility ratio **1.99×**, bias ≈ 0 — *better weather skill than
canola* (0.39), consistent with wheat yield tracking growing-season weather more directly.
Calibrated `rue` lands high (2.06, an effective value absorbing some structural
under-production) — the same pattern canola showed before its station set was expanded; adding
stations / refining canopy would bring it toward the agronomic 1.3–1.6 range.

**Sub-provincial validation (`scripts/validate_wheat_subprovincial.py`)** runs the same SK
RM-level check as canola, against the dashboard's "Spring Wheat" column (60-lb bushel).
Result — *opposite to canola*: local RM matching does **not** beat provincial
(local anomaly corr **0.34** vs provincial **0.41**). Wheat already has strong provincial-scale
weather skill, and single-RM wheat yields are noisier (variety, midge, FHB, protein-driven
management), so the provincial average is the cleaner target. Canola was the reverse — its weak
provincial signal meant local matching *added* skill. A genuine cross-crop difference.

## Sub-provincial validation (Saskatchewan)

[`subprovincial.py`](src/canola_dt/subprovincial.py) + `scripts/validate_subprovincial.py`
test the provincial-ceiling hypothesis directly, using real **RM-level (Rural Municipality)**
canola yields:

- **RM yields** — Saskatchewan Dashboard "RM Yields" export (SCIC + Crop Report), bu/ac → kg/ha.
- **RM locations** — Government of Saskatchewan ArcGIS feature service (298 RM centroids).

Each ECCC station is matched to its nearest RM; the station's *simulated* yield is compared to
that *local* RM's *observed* yield, with the same detrending as the provincial calibration.

```powershell
python scripts/validate_subprovincial.py
```

**Result (10 SK stations, 1995–2023, 287 station-years).** Interannual anomaly correlation,
same stations and method, only the target scale differs:

| yield target            | anomaly correlation |
|-------------------------|---------------------|
| provincial (StatCan SK) | +0.30               |
| **local RM (SCIC)**     | **+0.37**           |

Local matching beats provincial — confirming the hypothesis. The improvement is modest in the
pool because station-to-RM representativeness varies a lot: well-matched stations reach
**r = 0.5–0.66** (far above the provincial ceiling), while a few sit in RMs whose yields they
don't track (one weather *point* imperfectly represents even one RM). Point-in-polygon matching
gives the same pooled result (≈0.37), so the limiter is representativeness, not the join. The
next lever is **gridded weather per RM** (NASA POWER / ERA5) instead of a single station, and
extending to **Manitoba MASC** RM yields.

## Roadmap

- [x] Scaffold + synthetic end-to-end pipeline
- [x] Real ECCC weather ingestion (bulk daily, cached per station-year)
- [x] Historical yield join (StatCan) and model training (CV R² ≈ 0.59)
- [x] APSIM-style mechanistic crop model (phenology, RUE biomass, layered soil water, yield)
- [x] Calibrate process-model parameters vs trend-adjusted StatCan yields (level + pattern)
- [x] Multi-point simulation per province (10 stations each; offsets→≈0, corr plateaus ~0.39)
- [x] Sub-provincial validation vs SK SCIC RM-level yields (local corr 0.37 > provincial 0.30)
- [x] Couple process-model outputs (yield, biomass, LAI, water stress, timing) into ML features
- [x] Advisory layer: Canola Council agronomic alerts + calibrated process-model yield
- [x] Second crop: spring-wheat process model + calibration (anomaly corr 0.53 > canola 0.39)
- [x] Wheat sub-provincial validation vs SK RM spring-wheat (local 0.34 < provincial 0.41)
- [ ] Wheat advisory layer (Zadoks stages, FHB/midge timing, N-for-protein) from the wheat spec
- [ ] Gridded weather per RM (NASA POWER / ERA5) to fix station-vs-RM representativeness
- [ ] Extend sub-provincial validation to Manitoba MASC RM yields
- [ ] Sub-provincial (RM-level) ML model: local weather + process features vs SCIC yields
- [ ] Data assimilation step (Kalman/EnKF) to correct simulated state

## Key references

- Purcell, W., & Neubauer, T. (2023). *Digital Twins in Agriculture: A State-of-the-Art Review.*
- Kim, S., & Heo, ... (2024). *[Digital twin application in agriculture].*
