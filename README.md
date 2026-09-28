# Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**Smart India Hackathon — Problem Statement ID 26080**
Ministry of Earth Sciences (MoES) / NCMRWF

Raw numerical weather prediction (NWP) rainfall forecasts carry systematic errors that change with the weather situation: a model that under-predicts orographic rain behaves very differently from one that smooths out a monsoon depression. This project identifies the current weather **regime** as a soft probability vector, uses it to condition a neural bias-correction model, and returns a corrected rainfall forecast together with uncertainty, an explanation, and a verification report.

---

## Pipeline

```
Atmospheric predictors (u850, v850, q850, mslp anomaly, CAPE)
        │
        ▼
   Conv-VAE ──► latent space ──► GMM ──► soft regime probabilities
        │                                      │
        │                          Markov smoothing, regime-switch
        │                          forecast, novelty detection
        ▼                                      │
   Raw NWP rainfall ──► FiLM-conditioned U-Net ◄┘
                              │
                              ▼
              Corrected rainfall + quantiles (P10 / P50 / P90)
                              │
                              ▼
         Monte Carlo Dropout ──► P(rainfall ≥ threshold) maps
                              │
                              ▼
      Confidence-aware blend of AI and raw NWP forecast
                              │
                              ▼
   SHAP on an interpretable surrogate ──► plain-language explanation
                              │
                              ▼
   RMSE / POD / FAR / CSI / ETS / FSS report  +  Streamlit dashboard
```

## Key features

- **Soft weather regimes.** A convolutional VAE and a Gaussian Mixture Model produce a probability vector over five regimes (Active Monsoon, Break Monsoon, Monsoon Depression, Orographic Rainfall, Western Disturbance) rather than a hard label.
- **Regime-conditioned bias correction.** A U-Net with FiLM conditioning adapts its correction to the regime mix. Training uses a heavy-rainfall-weighted pinball loss to focus on extreme events.
- **Uncertainty.** Quantile heads plus 50-pass Monte Carlo Dropout give exceedance-probability maps for heavy and very heavy rainfall.
- **Temporal regime dynamics.** A learned Markov transition matrix smooths daily regime probabilities, estimates the chance that the current regime ends within N days, flags days that match no known regime, and retrieves similar historical days.
- **Confidence-aware blending.** When the regime call is uncertain or novel, the corrected forecast is blended back toward the raw NWP forecast.
- **Explainability.** SHAP on an interpretable surrogate model shows which synoptic indices (winds, moisture, pressure, CAPE) drove the regime call.
- **Verification.** RMSE, POD, FAR, CSI, ETS and Fractions Skill Score at IMD-style thresholds (15 / 65 / 115 mm).
- **Dashboard.** A Streamlit app for non-technical users, with technical detail on demand.

## Quick start

```bash
git clone https://github.com/Bhuvana908/sih_project.git
cd sih_project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train.py                   # generate data, train all models, evaluate
streamlit run dashboard/app.py    # launch the dashboard
```

`train.py` runs on CPU and prints a raw-vs-corrected verification table. It also saves everything the dashboard needs to `models/` and `outputs/`, so you only need to retrain when you change the config or the data.

Single-day inference from the command line:

```bash
python predict.py --day 10
```

Run the smoke tests:

```bash
python -m pytest tests/ -q
```

## Project structure

```
sih_project/
├── config.yaml              # hyperparameters, thresholds, paths
├── requirements.txt
├── train.py                 # end-to-end training and evaluation
├── predict.py               # single-day inference (CLI, also used by dashboard)
├── src/
│   ├── data_generation.py   # synthetic ERA5 / NWP / IMD-like data generator
│   ├── regime_model.py      # Conv-VAE + GMM
│   ├── regime_dynamics.py   # Markov smoothing, switch forecast, novelty, analogs
│   ├── bias_correction.py   # FiLM U-Net with quantile heads
│   ├── uncertainty.py       # MC Dropout and exceedance probabilities
│   ├── confidence_blend.py  # confidence-gated blend of AI and raw forecast
│   ├── explainability.py    # SHAP surrogate and readable explanations
│   ├── verification.py      # RMSE / POD / FAR / CSI / ETS / FSS
│   └── impact_index.py      # impact-based alert level (standalone module)
├── dashboard/
│   ├── app.py               # Streamlit dashboard
│   └── assets/              # logos
├── tests/
│   └── test_pipeline.py     # smoke tests
├── data/sample/             # cached dataset (created on first run)
├── models/                  # saved weights (created by train.py)
└── outputs/                 # predictions and verification report (created by train.py)
```

## Configuration

All settings live in `config.yaml`: grid size, number of regimes, VAE / GMM / U-Net hyperparameters, MC Dropout passes, quantiles, heavy-rain thresholds, and output paths. If you change data-related settings, delete `data/sample/synthetic_dataset.npz` and rerun `python train.py`.

## Data

The required real datasets are ERA5 reanalysis (Copernicus CDS), NCMRWF raw NWP forecasts, and IMD gridded observed rainfall. The last two are institutional and not publicly downloadable, so this repository ships a **physically motivated synthetic generator** (`src/data_generation.py`). Each regime has its own rainfall pattern, NWP bias signature and predictor signature, with persistent, smoothly transitioning regime sequences.

This lets the whole pipeline run end to end. It does **not** mean the reported scores will carry over unchanged to real data.

### Using real data

Replace the generator with a loader that returns arrays in these shapes:

| Array | Shape | Contents |
|---|---|---|
| `predictors` | `(N_days, 5, H, W)` | ERA5: `u850, v850, q850, mslp_anom, cape` |
| `raw_nwp` | `(N_days, H, W)` | NCMRWF raw rainfall forecast |
| `truth` | `(N_days, H, W)` | IMD gridded observed rainfall |

Optional sub-daily inputs (`subdaily_predictors`, shape `(N_days, steps_per_day, 5, H, W)`) power the dashboard's intra-day regime-shift panel. Keep `H == W` and set `domain.grid_size` to match.

## Mapping to the problem statement

| Requirement | Where |
|---|---|
| Identify the current weather regime | `src/regime_model.py` |
| Probabilistic regimes, not hard switches | GMM `predict_proba`, `src/regime_dynamics.py` |
| Regime-conditioned bias correction | `src/bias_correction.py` |
| Better heavy and very heavy rainfall forecasts | Weighted pinball loss, IMD thresholds in `src/verification.py` |
| Uncertainty and probability of extreme rain | `src/uncertainty.py` |
| Explainability | `src/explainability.py` |
| User-friendly district-level product | `dashboard/app.py` |
| Verification report | `src/verification.py`, `outputs/verification_report.json` |

## Limitations

- **Synthetic data.** Results demonstrate the method, not operational skill. Validation on real ERA5, NCMRWF and IMD data is the next step.
- **Surrogate explanations.** SHAP runs on a surrogate model that mimics the VAE + GMM regime probabilities, because the GMM step is not differentiable and the image input is high-dimensional. Check the surrogate's agreement with the real pipeline on your own data.
- **Coarse grid.** The demo domain is 24 × 24, not an operational district-resolution grid. Scaling up means increasing `domain.grid_size` and model capacity.
- **Impact index is a proof of concept.** `src/impact_index.py` combines exceedance probability with a vulnerability layer, but the layer included is a placeholder. Replace it with real data (population density, flood history, drainage) before drawing conclusions.

## Tech stack

Python, PyTorch, scikit-learn, SHAP, SciPy, NumPy, pandas, Matplotlib, Streamlit.
