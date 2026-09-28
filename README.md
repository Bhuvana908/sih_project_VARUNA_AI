# Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts
**SIH Problem Statement ID 26080 — Ministry of Earth Sciences (MoES) / NCMRWF**

A working, end-to-end implementation of the "probabilistic weather-regime" pipeline:

```
Atmospheric predictors (u850, v850, q850, mslp, CAPE)
        │
        ▼
 Conv-VAE  ──►  latent space  ──►  GMM  ──►  soft regime probability vector
   "Weather Mood Ring" (dimmer switch, not a hard label)
        │
        ▼  (FiLM conditioning)
 Raw NWP rainfall  ──►  FiLM-conditioned U-Net  ──►  Corrected rainfall (+ quantiles)
   "Smart Editor"
        │
        ▼
 Monte Carlo Dropout (50 passes)  ──►  P(rainfall ≥ threshold) maps
   "Probability Map"
        │
        ▼
 SHAP surrogate on interpretable synoptic indices  ──►  forecaster-readable explanation
   "The Detective"
        │
        ▼
 RMSE / POD / FAR / CSI / ETS / FSS verification report + Streamlit dashboard
```

## 1. Quick start

```bash
python -m venv venv && source venv/bin/activate       # optional but recommended
pip install -r requirements.txt

python train.py                     # generates data (first run), trains everything, evaluates
streamlit run dashboard/app.py      # interactive district-style dashboard
```

`train.py` takes a few minutes on CPU. It prints a verification table like:

```
Metric           Raw NWP    AI-Corrected
RMSE              12.548           4.885
CSI_65mm           0.067           0.518
ETS_65mm           0.015           0.463
FSS_65mm           0.300           0.923
...
```

showing the AI-corrected forecast substantially beating the raw NWP forecast, especially
at the heavy-rainfall thresholds the problem statement cares about most.

Everything needed by the dashboard is saved to `models/` and `outputs/` by `train.py`, so
you only need to re-run training when you change the config or the data.

## 2. Why the shipped data is synthetic

Real inputs for this problem are:
- **ERA5 reanalysis** (atmospheric predictors) — needs a free Copernicus CDS API key and
  multi-GB downloads per season.
- **NCMRWF raw NWP rainfall forecasts** and **IMD gridded observed rainfall** — both are
  institutional datasets not available for public/automated download.

None of that is reachable from this environment, so `src/data_generation.py` ships a
**physically-motivated synthetic generator**: each of the 5 weather regimes named in the
problem statement (Active Monsoon, Break Monsoon, Monsoon Depression, Orographic Rainfall,
Western Disturbance) has its own characteristic rainfall spatial pattern, its own NWP bias
signature (e.g. NWP badly under-resolving orographic rain, or smoothing out a depression's
peak intensity), and its own synoptic predictor signature (winds/moisture/pressure/CAPE) —
with realistic day-to-day regime persistence and **smooth transitions** between regimes
(not hard switches), so the GMM genuinely has to output fractional probabilities near
transition days rather than trivial 0%/100% labels. The regime mix is also weighted
(`data.regime_prior` in the generator) so Active/Break/Orographic monsoon-core regimes
dominate, Monsoon Depression appears as an occasional high-impact event, and Western
Disturbance — a predominantly winter phenomenon — occurs only rarely, matching real
monsoon-season climatology rather than appearing as often as the core regimes. You can
verify this yourself:

```bash
python -c "
import numpy as np
d = np.load('outputs/test_results.npz')
print(d['regime_probs'][:15].round(2))"
```

This makes the *whole pipeline* (VAE → GMM → FiLM U-Net → MC-Dropout → SHAP → verification)
runnable and demonstrably improving on a raw forecast, which is what matters for building
and defending the idea at SIH. It is not a claim that the reported skill scores would
transfer unchanged to real IMD/NCMRWF data — real verification requires real data.

### Using real data
Swap `src/data_generation.py` for a loader that reads:
- `predictors`: `(N_days, 5, H, W)` array of ERA5 fields (`u850,v850,q850,mslp_anom,cape`),
  e.g. read via `xarray`/`cdsapi` and regridded to your chosen domain.
- `raw_nwp`: `(N_days, H, W)` array of NCMRWF raw rainfall forecast, same grid.
- `truth`: `(N_days, H, W)` array of IMD gridded observed rainfall, same grid.

Keep the same array shapes/dtypes and everything downstream (`train.py`, `predict.py`,
the dashboard) works unchanged. `config.yaml` -> `domain.grid_size` should match your `H=W`
(use a square regridded domain, or generalize the U-Net's pooling if `H != W`).

**Sub-daily transitions.** `generate_dataset()` also returns `subdaily_predictors`
(`N_days, steps_per_day, 5, H, W`) and `subdaily_regime` (validation-only labels), used by
the dashboard's "When during the day might this shift happen?" panel. With real data, this
is just ERA5 pulled at its native 6-hourly (or finer) resolution instead of one reading per
day — `raw_nwp`/`truth` stay daily, since IMD/NCMRWF gridded products are day-resolution.
`config.yaml` -> `subdaily.steps_per_day` / `subdaily.wobble_std` control the synthetic
version only and are unused once real sub-daily ERA5 data is supplied.

## 3. Project layout

```
config.yaml                 all hyperparameters & paths
requirements.txt
train.py                    end-to-end training + evaluation script
predict.py                  single-day inference (CLI) — also used by the dashboard
src/
  data_generation.py        synthetic ERA5/NWP/IMD-like data generator
  regime_model.py           Conv-VAE + GMM ("Weather Mood Ring")
  bias_correction.py        FiLM-conditioned U-Net + quantile regression heads
  uncertainty.py            Monte Carlo Dropout + exceedance-probability maps
  explainability.py         SHAP surrogate model + human-readable explanations
  verification.py           RMSE / POD / FAR / CSI / ETS / FSS
dashboard/
  app.py                    Streamlit dashboard (3 maps + regime bars + SHAP + scores)
data/sample/                cached synthetic dataset (generated on first run)
models/                     saved model weights + fitted GMM/surrogate (after train.py)
outputs/                    cached test-set predictions + verification_report.json
```

## 4. How each SIH-brief requirement maps to code

| Brief requirement | Implementation |
|---|---|
| Identify the current weather regime | `src/regime_model.py` (Conv-VAE + GMM) |
| Probabilistic / soft regimes, not hard switch | GMM `predict_proba`, smoothed regime transitions in `data_generation.py` |
| Regime-conditioned bias correction | `src/bias_correction.py` FiLM U-Net |
| Improve heavy/very heavy rainfall forecasts specifically | heavy-rain-weighted pinball loss (`heavy_rain_weighted_pinball`) + IMD thresholds in verification |
| Uncertainty / probability of extreme rain | `src/uncertainty.py` (MC Dropout + quantile heads) |
| Explainability | `src/explainability.py` (SHAP on interpretable surrogate) |
| District-level, user-friendly product | `dashboard/app.py` Streamlit dashboard |
| Verification report (RMSE, ETS, CSI, ...) | `src/verification.py`, `outputs/verification_report.json` |

## 5. Re-running with different settings

Edit `config.yaml` (grid size, number of synthetic seasons, epochs, thresholds, MC-Dropout
passes, etc.), delete the cached `data/sample/synthetic_dataset.npz` if you changed the data
settings, and re-run `python train.py`.

## 6. Known limitations (be upfront about these when presenting)

- Data is synthetic (see §2) — real-data validation is the natural next step and is called
  out explicitly rather than hidden.
- The SHAP explanation runs on a surrogate model (interpretable summary features →
  regime probabilities) rather than directly on the Conv-VAE/GMM, because SHAP does not
  handle a non-differentiable GMM step or very high-dimensional image inputs well. The
  surrogate reproduces the real pipeline's regime probabilities with R² > 0.9 on all
  regimes and ~97% top-regime agreement in this demo (recompute for your own data with the
  snippet in `train.py`'s XAI section).
- Grid is a coarse 24×24 demo domain, not an operational district-resolution grid; scaling
  up mainly means increasing `domain.grid_size` and model capacity in `config.yaml`.
