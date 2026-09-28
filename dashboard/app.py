"""
Streamlit dashboard for the Regime-Aware AI Post-Processing pipeline.
Redesigned for a non-technical audience (disaster-management officials,
judges, general users) while still exposing technical detail on demand.

Run from the project root with:
    streamlit run dashboard/app.py
"""
import os
import base64
import sys
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import regime_model as rm
from src import bias_correction as bc
from src import uncertainty as unc
from src import explainability as ex
from src import verification as ver
from src import data_generation as dg
from src import regime_dynamics as rd
from src import confidence_blend as cb

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

st.set_page_config(page_title="VARUNA AI", layout="wide",
                   page_icon=os.path.join(ASSETS_DIR, "varuna_icon.png"))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------
# Plain-language reference tables (kept in one place so they're easy to edit)
# ----------------------------------------------------------------------
REGIME_DESCRIPTIONS = {
    "Active Monsoon": "Widespread, heavy rain likely across most of the region.",
    "Break Monsoon": "A dry spell — little to no rain expected in most areas.",
    "Monsoon Depression": "An intense, localized storm system — high flood risk in a small area.",
    "Orographic Rainfall": "Heavy rain concentrated near hills/mountains (e.g. Western Ghats).",
    "Western Disturbance": "A rarer weather system occasionally influencing rainfall outside the core monsoon pattern.",
}
FRIENDLY_VAR_NAMES = {
    "u850": "East-West wind",
    "v850": "North-South wind",
    "q850": "Atmospheric moisture",
    "mslp_anom": "Air pressure anomaly",
    "cape": "Storm energy (CAPE)",
}
VAR_UNITS = {
    "u850": "m/s", "v850": "m/s", "q850": "g/kg", "mslp_anom": "hPa", "cape": "J/kg",
}
REGION_PHRASES = {
    "domain_mean": "across the region",
    "NW": "in the northwest",
    "NE": "in the northeast",
    "SW": "in the southwest",
    "SE": "in the southeast",
}
METRIC_HELP = {
    "RMSE": "Typical size of the forecast error, in mm. Lower is better.",
    "POD": "Of the heavy-rain events that really happened, what % did the forecast catch? Higher is better.",
    "FAR": "Of the times heavy rain was forecast, what % turned out to be a false alarm? Lower is better.",
    "CSI": "Overall accuracy at catching real heavy-rain events without too many false alarms. Higher is better.",
    "ETS": "Same idea as CSI, adjusted so pure luck can't inflate the score. Higher is better.",
    "FSS": "How well the forecast gets the *location* of heavy rain right, not just the amount. Higher is better.",
}

# A day is flagged as "transitioning" when the second-most-likely regime is
# still carrying a meaningful share of the probability mass, i.e. the
# atmosphere hasn't settled cleanly into one regime. This mirrors the blend
# logic already used in src/data_generation.py (w[secondary] > 0.15) to
# construct physically-realistic transition days.
TRANSITION_THRESHOLD = 0.20


def detect_regime_transition(regime_probs, regime_names, threshold=TRANSITION_THRESHOLD):
    """Flag whether this day's regime signal looks mixed/transitional rather
    than settled into one clear regime.

    Returns None if the day is "settled" (secondary regime below threshold),
    otherwise a dict describing the two competing regimes and their shares.
    """
    order = np.argsort(-regime_probs)
    top_idx, second_idx = order[0], order[1]
    top_p, second_p = float(regime_probs[top_idx]), float(regime_probs[second_idx])
    if second_p < threshold:
        return None
    return {
        "from_regime": regime_names[second_idx],
        "to_regime": regime_names[top_idx],
        "from_prob": second_p,
        "to_prob": top_p,
        "gap": top_p - second_p,
    }


def transition_banner(transition):
    """Render a plain-language banner describing a within-day regime transition."""
    close_call = transition["gap"] < 0.10
    msg = (
        f"**Transition signal on this day** — conditions look mixed between "
        f"**{transition['from_regime']}** ({transition['from_prob']*100:.0f}%) and "
        f"**{transition['to_regime']}** ({transition['to_prob']*100:.0f}%), "
        f"rather than settled cleanly into one regime."
    )
    if close_call:
        st.warning(msg + " The two are close enough that this could go either way.")
    else:
        st.info(msg)


# ----------------------------------------------------------------------
# Time-of-day transition explorer
#
# IMPORTANT CAVEAT: the underlying dataset has one predictor snapshot per
# calendar day, not real sub-daily (hourly) observations. There is no
# genuine diurnal signal to read a transition clock-time off of. What we
# build below is a smooth, physically-reasonable ESTIMATE: we linearly
# interpolate the atmospheric state between yesterday's snapshot, today's
# snapshot, and tomorrow's snapshot (treating each as local noon), then
# run the *same trained* regime classifier (VAE + GMM) on each interpolated
# hourly state. This turns a 3-point day-to-day sequence into a plausible
# hour-by-hour curve and lets us read off roughly *when* within the day the
# dominant regime label would flip — useful for exploring the model's
# behaviour, but it is an interpolation, not a measurement, and the
# dashboard says so explicitly.
# ----------------------------------------------------------------------
HOUR_STEP = 3  # sample every 3 hours across the 24h day


def classify_grid(assets, predictor_grid):
    """Run the trained VAE + GMM regime classifier on one predictor grid."""
    pmean, pstd = assets["pmean"], assets["pstd"]
    norm_grid = (predictor_grid[None] - pmean) / pstd
    latent = rm.get_latents(assets["vae"], norm_grid.astype(np.float32))[0]
    regime_probs_raw = assets["gmm"].predict_proba(latent[None])[0]
    return regime_probs_raw[assets["reorder"]]


def _interp_grid_at_hour(h, prev_grid, today_grid, next_grid):
    """Piecewise-linear interpolation of the predictor grid at hour h (0-24),
    treating yesterday/today/tomorrow's snapshots as anchored at noon
    (-12, 12, 36)."""
    if h <= 12:
        frac = (h - (-12)) / 24.0
        return prev_grid + frac * (today_grid - prev_grid)
    frac = (h - 12) / 24.0
    return today_grid + frac * (next_grid - today_grid)


@st.cache_data
def build_hourly_trajectory(_assets, day, n_test_days, predictors, regime_names, hour_step=HOUR_STEP):
    prev_day = max(day - 1, 0)
    next_day = min(day + 1, n_test_days - 1)
    prev_grid, today_grid, next_grid = predictors[prev_day], predictors[day], predictors[next_day]

    hours = np.arange(0, 24, hour_step)
    probs = np.zeros((len(hours), len(regime_names)))
    for i, h in enumerate(hours):
        grid_h = _interp_grid_at_hour(float(h), prev_grid, today_grid, next_grid)
        probs[i] = classify_grid(_assets, grid_h)
    return hours, probs


def find_hourly_crossings(hours, probs, regime_names):
    """Find approximate clock-times where the dominant (argmax) regime
    changes between consecutive sampled hours, via linear interpolation of
    the two competing regimes' probability curves."""
    dominant = probs.argmax(axis=1)
    crossings = []
    for i in range(1, len(hours)):
        r_from, r_to = dominant[i - 1], dominant[i]
        if r_from == r_to:
            continue
        diff0 = probs[i - 1, r_from] - probs[i - 1, r_to]
        diff1 = probs[i, r_from] - probs[i, r_to]
        denom = diff0 - diff1
        frac = 0.5 if abs(denom) < 1e-9 else float(np.clip(diff0 / denom, 0, 1))
        cross_hour = hours[i - 1] + frac * (hours[i] - hours[i - 1])
        crossings.append({
            "hour": cross_hour,
            "from_regime": regime_names[r_from],
            "to_regime": regime_names[r_to],
        })
    return crossings


def fmt_hour(h):
    hh = int(h) % 24
    mm = int(round((h - int(h)) * 60))
    return f"{hh:02d}:{mm:02d}"


def plot_regime_trajectory(hours, probs, regime_names, crossings):
    """Line plot of regime probability across the day, with the crossing
    point(s) marked explicitly — a plain multi-line chart makes two lines
    swapping rank hard to notice, especially when the crossing falls
    between two sampled hours rather than on one."""
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for i, name in enumerate(regime_names):
        ax.plot(hours, probs[:, i], marker="o", markersize=4, linewidth=2,
                label=name)

    for c in crossings:
        ax.axvline(c["hour"], color="gray", linestyle="--", linewidth=1, alpha=0.7)
        # interpolate the y-value at the crossing so the dot sits exactly on the lines
        y = np.interp(c["hour"], hours, probs[:, regime_names.index(c["to_regime"])])
        ax.plot(c["hour"], y, "o", color="black", markersize=6, zorder=5)
        ax.annotate(
            fmt_hour(c["hour"]), xy=(c["hour"], 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
            fontsize=9, color="dimgray",
        )

    ax.set_xticks(hours)
    ax.set_xticklabels([fmt_hour(h) for h in hours])
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("Regime probability")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def friendly_feature_name(feature_name):
    """'q850_NW' -> ('Atmospheric moisture', 'in the northwest', 'g/kg')"""
    for var in sorted(dg.PREDICTOR_CHANNELS, key=len, reverse=True):
        if feature_name.startswith(var + "_"):
            suffix = feature_name[len(var) + 1:]
            region = REGION_PHRASES.get(suffix, suffix)
            return FRIENDLY_VAR_NAMES.get(var, var), region, VAR_UNITS.get(var, "")
    return feature_name, "", ""


def build_plain_explanation(explanation, top_k=2):
    """Turn the raw SHAP top_features list into a short, plain-English paragraph."""
    regime = explanation["top_regime"]
    conf = explanation["top_regime_confidence"] * 100
    reasons = []
    for fname, impact, raw in explanation["top_features"][:top_k]:
        var, region, unit = friendly_feature_name(fname)
        direction = "supported" if impact > 0 else "worked against"
        reasons.append(f"**{var}** {region} (measured at {raw:.1f}{unit}), which {direction} this call")
    reason_text = " and ".join(reasons)
    return (
        f"The AI thinks today looks like a **{regime}** day ({conf:.0f}% confidence). "
        f"The main signals were: {reason_text}."
    )


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_assets(_cfg):
    models_dir = os.path.join(PROJECT_ROOT, _cfg["paths"]["models_dir"])
    grid = _cfg["domain"]["grid_size"]
    n_regimes = _cfg["domain"]["n_regimes"]
    import torch

    norm = np.load(f"{models_dir}/vae_norm.npz")
    pmean, pstd = norm["mean"], norm["std"]

    vae = rm.ConvVAE(len(dg.PREDICTOR_CHANNELS), grid, _cfg["vae"]["latent_dim"])
    vae.load_state_dict(torch.load(f"{models_dir}/vae.pt", map_location="cpu"))
    vae.eval()

    with open(f"{models_dir}/gmm.pkl", "rb") as f:
        gmm_data = pickle.load(f)
    gmm, cluster_to_name_idx = gmm_data["gmm"], gmm_data["cluster_to_name_idx"]
    reorder = [k for k, v in sorted(cluster_to_name_idx.items(), key=lambda kv: kv[1])]

    quantiles = tuple(_cfg["uncertainty"]["quantiles"])
    unet = bc.FiLMUNet(in_ch=1, cond_dim=n_regimes, base=_cfg["unet"]["base_channels"],
                        dropout_p=_cfg["unet"]["dropout_p"], quantiles=quantiles)
    unet.load_state_dict(torch.load(f"{models_dir}/unet.pt", map_location="cpu"))
    unet.eval()

    with open(f"{models_dir}/surrogate.pkl", "rb") as f:
        surr_data = pickle.load(f)

    rdyn_path = f"{models_dir}/regime_dynamics.npz"
    rdyn = np.load(rdyn_path) if os.path.exists(rdyn_path) else None

    return {
        "vae": vae, "pmean": pmean, "pstd": pstd, "gmm": gmm, "reorder": reorder,
        "unet": unet, "quantiles": quantiles,
        "surrogate": surr_data["model"], "feature_names": surr_data["feature_names"],
        "transition_matrix": rdyn["transition_matrix"] if rdyn is not None else None,
        "novelty_threshold": float(rdyn["novelty_threshold"]) if rdyn is not None else None,
        "train_latents": rdyn["train_latents"] if rdyn is not None else None,
        "train_hard_regime_named": rdyn["train_hard_regime_named"] if rdyn is not None else None,
    }


@st.cache_data
def load_test_results(_cfg):
    outputs_dir = os.path.join(PROJECT_ROOT, _cfg["paths"]["outputs_dir"])
    npz = np.load(f"{outputs_dir}/test_results.npz")
    with open(f"{outputs_dir}/verification_report.json") as f:
        report = json.load(f)
    return {k: npz[k] for k in npz.files}, report


@st.cache_data
def build_scenarios(_cfg, test_data_hash, truth, regime_probs, regime_names):
    """Pick a small set of illustrative, human-labelled days instead of making
    the user guess a day number. Computed from the data so it stays valid if
    the dataset is regenerated."""
    top = regime_probs.argmax(axis=1)
    day_max = truth.max(axis=(1, 2))
    scenarios = {}

    wettest = int(np.argmax(day_max))
    scenarios[f"Most extreme rainfall day (Day {wettest})"] = wettest

    calmest = int(np.argmin(truth.mean(axis=(1, 2))))
    scenarios[f"Calm / low-rainfall day (Day {calmest})"] = calmest

    for i, name in enumerate(regime_names):
        idxs = np.where(top == i)[0]
        if len(idxs) == 0:
            continue
        best = int(idxs[np.argmax(day_max[idxs])])
        scenarios[f"{name} example (Day {best})"] = best

    # Most ambiguous day: smallest gap between the top two regime
    # probabilities, i.e. the clearest example of a within-day transition.
    sorted_probs = -np.sort(-regime_probs, axis=1)
    gaps = sorted_probs[:, 0] - sorted_probs[:, 1]
    transition_day = int(np.argmin(gaps))
    if gaps[transition_day] < TRANSITION_THRESHOLD:
        scenarios[f"Regime transition example (Day {transition_day})"] = transition_day

    return scenarios


@st.cache_data
def compute_regime_breakdown(_cfg, test_data_hash, raw_nwp, corrected, truth, regime_probs, regime_names):
    """Splits the test-set verification metrics out by each day's dominant
    regime, so the aggregate scores in the sidebar don't hide regimes where
    the correction helps a lot more (or less) than average."""
    dominant = regime_probs.argmax(axis=1)
    thresholds = _cfg["verification"]["thresholds_mm"]
    neighborhood = _cfg["verification"]["fss_neighborhood"]
    raw_by_regime = ver.per_regime_report(raw_nwp, truth, dominant, len(regime_names),
                                           regime_names, thresholds, neighborhood)
    corr_by_regime = ver.per_regime_report(corrected, truth, dominant, len(regime_names),
                                            regime_names, thresholds, neighborhood)
    return raw_by_regime, corr_by_regime


def run_for_day(assets, predictor_grid, raw_nwp_grid, cfg):
    pmean, pstd = assets["pmean"], assets["pstd"]
    norm_grid = (predictor_grid[None] - pmean) / pstd
    latent = rm.get_latents(assets["vae"], norm_grid.astype(np.float32))[0]
    regime_probs_raw = assets["gmm"].predict_proba(latent[None])[0]
    regime_probs = regime_probs_raw[assets["reorder"]]

    unet_out = unc.predict_with_uncertainty(
        assets["unet"], raw_nwp_grid, regime_probs, assets["quantiles"],
        n_mc_passes=cfg["uncertainty"]["mc_dropout_passes"],
        heavy_threshold=cfg["uncertainty"]["heavy_rain_threshold_mm"],
    )
    X_row, _ = ex.build_feature_matrix(predictor_grid[None], dg.PREDICTOR_CHANNELS)
    explanation = ex.explain_day(assets["surrogate"], X_row[0], assets["feature_names"],
                                  cfg["domain"]["regime_names"])

    dynamics = None
    if assets["transition_matrix"] is not None:
        trans = assets["transition_matrix"]
        is_novel, novelty_score = rd.is_novel_regime(assets["gmm"], latent, assets["novelty_threshold"])
        dynamics = {
            "switch_in_2_days_prob": rd.probability_of_regime_switch(regime_probs, trans, horizon_days=2),
            "expected_switch_day": rd.days_until_likely_switch(regime_probs, trans),
            "is_novel_regime": is_novel,
            "novelty_score": novelty_score,
            "analog_days": rd.find_analog_days(
                latent, assets["train_latents"], assets["train_hard_regime_named"],
                cfg["domain"]["regime_names"], k=3,
            ),
        }

    n_regimes = len(cfg["domain"]["regime_names"])
    is_novel_for_conf = dynamics["is_novel_regime"] if dynamics is not None else False
    confidence = cb.effective_confidence(regime_probs, n_regimes, is_novel_for_conf)
    blended_forecast = cb.blend_forecast(raw_nwp_grid, unet_out["quantile_maps"]["q50"], confidence)

    return regime_probs, unet_out, explanation, dynamics, confidence, blended_forecast


def plot_grid(field, title, cmap="Blues", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def regime_bar(name, prob, is_top):
    prob = float(prob)
    pct = int(round(prob * 100))
    weight = "**" if is_top else ""
    st.markdown(f"{weight}{name} — {pct}%{weight}")
    st.progress(min(max(prob, 0.0), 1.0))
    if is_top:
        st.caption(REGIME_DESCRIPTIONS.get(name, ""))


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------
def main():
    cfg = load_config()

    with open(os.path.join(ASSETS_DIR, "varuna_logo.png"), "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("ascii")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:18px;margin-bottom:4px;">'
        f'<img src="data:image/png;base64,{logo_b64}" alt="VARUNA AI logo" '
        f'style="height:72px;width:72px;flex:none;object-fit:contain;display:block;" />'
        f'<h1 style="margin:0;padding:0;line-height:1.1;letter-spacing:2px;">VARUNA AI</h1>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Regime-aware AI post-processing of monsoon rainfall forecasts. See how AI improves "
        "an official rainfall forecast, region by region, and how confident it is about heavy-rain risk."
    )

    models_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["models_dir"])
    if not os.path.exists(f"{models_dir}/unet.pt"):
        st.error("No trained models found. Run `python train.py` from the project root first.")
        st.stop()

    assets = load_assets(cfg)
    test_data, verification_report = load_test_results(cfg)
    n_test_days = test_data["raw_nwp"].shape[0]
    regime_names = cfg["domain"]["regime_names"]

    scenarios = build_scenarios(cfg, n_test_days, test_data["truth"],
                                 test_data["regime_probs"], regime_names)
    scenario_labels = list(scenarios.keys()) + ["Browse a specific day"]

    with st.sidebar:
        st.header("Choose what to look at")
        choice = st.selectbox("Pick an example day", scenario_labels, index=0)
        if choice == "Browse a specific day":
            day = st.slider("Day", 0, n_test_days - 1, 0)
        else:
            day = scenarios[choice]

        st.markdown("---")
        compare_enabled = False
        compare_day = day
        if day > 0:
            compare_enabled = st.checkbox("Compare with an earlier day")
            if compare_enabled:
                compare_day = st.slider("Compare against day (must be before today)",
                                         0, day - 1, max(day - 3, 0))
        else:
            st.caption("No earlier day available to compare (this is the first day in the test set).")

        st.markdown("---")
        st.subheader("Overall accuracy (all test days)")
        st.caption("Comparing the AI-corrected forecast to the original raw forecast.")
        raw_r = verification_report["raw_nwp"]
        corr_r = verification_report["ai_corrected"]
        df = pd.DataFrame({"Raw forecast": raw_r, "AI-corrected": corr_r})
        st.dataframe(df.style.format("{:.3f}"), height=420)
        with st.expander("What do these metrics mean?"):
            for k, v in METRIC_HELP.items():
                st.markdown(f"**{k}** — {v}")

        with st.expander("Accuracy by regime"):
            st.caption(
                "The table above is averaged across all test days. This breaks it out by each "
                "day's dominant regime — a heavy threshold (65mm) is more forgiving to hit by "
                "chance on some regimes than others, so the aggregate score alone can hide that."
            )
            raw_by_regime, corr_by_regime = compute_regime_breakdown(
                cfg, n_test_days, test_data["raw_nwp"], test_data["corrected"],
                test_data["truth"], test_data["regime_probs"], regime_names,
            )
            rows = []
            for name in regime_names:
                if name not in corr_by_regime:
                    continue
                raw_m = raw_by_regime[name]["report"]
                corr_m = corr_by_regime[name]["report"]
                rows.append({
                    "Regime": name,
                    "Test days": raw_by_regime[name]["n_days"],
                    "RMSE (raw)": raw_m["RMSE"],
                    "RMSE (AI)": corr_m["RMSE"],
                    "CSI_65mm (raw)": raw_m.get("CSI_65mm", float("nan")),
                    "CSI_65mm (AI)": corr_m.get("CSI_65mm", float("nan")),
                })
            if rows:
                breakdown_df = pd.DataFrame(rows).set_index("Regime")
                numeric_cols = ["RMSE (raw)", "RMSE (AI)", "CSI_65mm (raw)", "CSI_65mm (AI)"]
                st.dataframe(breakdown_df.style.format({c: "{:.2f}" for c in numeric_cols}))
                st.caption("Regimes with fewer than 3 test days are omitted — too few to report reliably.")
            else:
                st.caption("Not enough test days per regime to break this out (need at least 3 per regime).")

    predictor_grid = test_data["predictors"][day]
    raw_nwp_grid = test_data["raw_nwp"][day]
    truth_grid = test_data["truth"][day]

    with st.spinner("Analyzing today's atmosphere and correcting the forecast..."):
        regime_probs, unet_out, explanation, dynamics, confidence, blended_forecast = run_for_day(
            assets, predictor_grid, raw_nwp_grid, cfg)

    vmax = max(raw_nwp_grid.max(), truth_grid.max(), unet_out["quantile_maps"]["q50"].max()) * 1.05

    # ---------------- Rainfall maps ----------------
    st.subheader("Rainfall: before and after AI correction")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(plot_grid(raw_nwp_grid, "Original forecast (mm)", vmin=0, vmax=vmax))
        st.caption("What the standard weather model predicted, before any correction.")
    with c2:
        st.pyplot(plot_grid(unet_out["quantile_maps"]["q50"], "AI-corrected forecast (mm)", vmin=0, vmax=vmax))
        st.caption("The same forecast after the AI adjusts it for today's weather pattern.")
    with c3:
        st.pyplot(plot_grid(truth_grid, "What actually happened (mm)", vmin=0, vmax=vmax))
        st.caption("Ground-truth rainfall, for comparison.")

    # ---------------- Confidence-aware recommended forecast ----------------
    st.subheader("Recommended forecast (confidence-aware)")
    st.caption(
        "The AI-corrected map above is always shown in full. This is different: it only trusts "
        "the correction as much as the regime call underneath it deserves, and leans back toward "
        "the original forecast when that call is shaky."
    )
    rc1, rc2 = st.columns([1, 1])
    with rc1:
        st.pyplot(plot_grid(blended_forecast, "Recommended forecast (mm)", vmin=0, vmax=vmax))
        st.caption("A blend of the raw and AI-corrected forecasts, weighted by correction confidence.")
    with rc2:
        st.metric("Correction confidence", f"{confidence*100:.0f}%",
                   help="0% = regime call was no better than a random guess (or flagged as novel); "
                        "100% = the regime classifier was fully certain.")
        if confidence >= 0.8:
            st.success("High confidence — the recommended forecast is close to the full AI correction.")
        elif confidence >= 0.4:
            st.info("Moderate confidence — the recommended forecast is a genuine blend of both.")
        else:
            st.warning("Low confidence — the recommended forecast leans mostly on the raw NWP forecast.")

    # ---------------- Flood risk ----------------
    st.subheader(f"Flood risk outlook — chance of over {cfg['uncertainty']['heavy_rain_threshold_mm']:.0f} mm of rain")
    c4, c5 = st.columns([1, 1])
    with c4:
        st.pyplot(plot_grid(unet_out["exceedance_probability"], "Chance of extreme rainfall",
                             cmap="Reds", vmin=0, vmax=1))
        st.caption("Darker red = higher chance of dangerously heavy rain in that area. "
                    "Mostly white is a good sign — it means low risk that day.")
    with c5:
        st.pyplot(plot_grid(unet_out["mc_std"], "How uncertain is the AI?", cmap="Purples"))
        st.caption("Darker purple = the AI is less sure of its forecast there — "
                    "often near intense, fast-changing storms.")

    # ---------------- Regime identification ----------------
    st.subheader("What kind of monsoon day is this?")
    st.caption(
        "Instead of forcing today into a single fixed category, the AI gives a confidence "
        "breakdown across known weather patterns — because real weather often blends between them."
    )
    transition = detect_regime_transition(regime_probs, regime_names)
    if transition is not None:
        transition_banner(transition)

    with st.expander("When during the day might this shift happen?"):
        has_real_subdaily = "subdaily_regime_probs" in test_data
        if has_real_subdaily:
            hours = test_data["subdaily_hours"]
            hourly_probs = test_data["subdaily_regime_probs"][day]
        else:
            st.caption(
                "The loaded outputs/test_results.npz predates sub-daily support, so this is "
                "a smooth, interpolated *estimate* between yesterday's, today's and "
                "tomorrow's snapshot rather than a real sub-daily reading. Re-run "
                "`python train.py` (after pulling the updated code) to get real sub-daily "
                "model output here."
            )
            hours, hourly_probs = build_hourly_trajectory(
                assets, day, n_test_days, test_data["predictors"], regime_names,
            )

        crossings = find_hourly_crossings(hours, hourly_probs, regime_names)
        st.pyplot(plot_regime_trajectory(hours, hourly_probs, regime_names, crossings))
        if crossings:
            st.caption("Dashed line(s) mark where the dominant regime crosses over.")

        if crossings:
            st.markdown("**Shift times:**" if has_real_subdaily else "**Estimated shift times:**")
            for c in crossings:
                st.markdown(
                    f"- ~**{fmt_hour(c['hour'])}** — {c['from_regime']} "
                    f"to {c['to_regime']}"
                )
        else:
            st.caption("No dominant-regime crossover across the day — conditions look steady.")

    ccols = st.columns([1.1, 1.4])
    with ccols[0]:
        order = np.argsort(-regime_probs)
        top_idx = order[0]
        for i in order:
            regime_bar(regime_names[i], regime_probs[i], is_top=(i == top_idx))

    with ccols[1]:
        st.markdown("**Why the AI thinks this:**")
        st.info(build_plain_explanation(explanation))
        with st.expander("Show technical details"):
            feat_df = pd.DataFrame(explanation["top_features"],
                                    columns=["Feature", "SHAP impact", "Value"])
            st.dataframe(feat_df, hide_index=True)
            st.caption(
                "SHAP values quantify how much each atmospheric variable pushed the model's "
                "decision, in the units the model was trained on. Positive = pushed toward the "
                "predicted regime; negative = pushed away from it."
            )

    # ---------------- Regime outlook (persistence + switch forecast) ----------------
    st.subheader("Regime outlook")
    if dynamics is None:
        st.warning(
            "No regime-dynamics data found in models/. Re-run `python train.py` "
            "after pulling the updated code to see the switch forecast, novelty flag, "
            "and analog days here."
        )
    else:
        oc1, oc2 = st.columns(2)
        with oc1:
            st.metric("Chance current regime ends within 2 days",
                      f"{dynamics['switch_in_2_days_prob']*100:.0f}%")
        with oc2:
            if dynamics["expected_switch_day"] is not None:
                st.metric("Expected time to next regime change", f"~{dynamics['expected_switch_day']} day(s)")
            else:
                st.metric("Expected time to next regime change", "> 7 days")
        st.markdown("**Closest historical analog days:**")
        for a in dynamics["analog_days"]:
            st.markdown(
                f"- Day {a['day_index']}: **{a['regime_name']}**, "
                f"which then persisted **{a['days_persisted_after']} more day(s)**"
            )

    # ---------------- Day comparison: what changed and why ----------------
    if compare_enabled and compare_day != day:
        st.markdown("---")
        st.subheader(f"What changed: Day {compare_day} to Day {day}")

        predictor_grid_b = test_data["predictors"][compare_day]
        raw_nwp_grid_b = test_data["raw_nwp"][compare_day]
        with st.spinner("Analyzing the comparison day..."):
            (regime_probs_b, unet_out_b, explanation_b,
             dynamics_b, confidence_b, blended_b) = run_for_day(
                assets, predictor_grid_b, raw_nwp_grid_b, cfg)

        shared_vmax = max(unet_out["quantile_maps"]["q50"].max(),
                           unet_out_b["quantile_maps"]["q50"].max()) * 1.05

        dcol1, dcol2 = st.columns(2)
        panels = [
            (dcol1, compare_day, regime_probs_b, unet_out_b, explanation_b),
            (dcol2, day, regime_probs, unet_out, explanation),
        ]
        for col, d_idx, r_probs, u_out, expl in panels:
            with col:
                top_i = int(np.argmax(r_probs))
                st.markdown(f"**Day {d_idx}**")
                st.markdown(f"**{regime_names[top_i]}** — {r_probs[top_i]*100:.0f}% confidence")
                st.pyplot(plot_grid(u_out["quantile_maps"]["q50"], "AI-corrected (mm)",
                                     vmin=0, vmax=shared_vmax))
                st.caption(build_plain_explanation(expl, top_k=1))

        st.markdown("**Summary of what changed:**")
        top_a = int(np.argmax(regime_probs_b))
        top_b = int(np.argmax(regime_probs))
        rain_a = float(unet_out_b["quantile_maps"]["q50"].mean())
        rain_b = float(unet_out["quantile_maps"]["q50"].mean())

        if top_a == top_b:
            st.markdown(
                f"- Regime stayed **{regime_names[top_b]}** on both days "
                f"(confidence {regime_probs_b[top_a]*100:.0f}% to {regime_probs[top_b]*100:.0f}%)."
            )
        else:
            st.markdown(f"- Regime shifted from **{regime_names[top_a]}** to **{regime_names[top_b]}**.")

        rain_delta = rain_b - rain_a
        st.markdown(
            f"- AI-corrected average rainfall: {rain_a:.1f}mm to {rain_b:.1f}mm "
            f"({'+' if rain_delta >= 0 else ''}{rain_delta:.1f}mm)"
        )
        st.markdown(f"- Correction confidence: {confidence_b*100:.0f}% to {confidence*100:.0f}%")

        feats_a = explanation_b.get("top_features", [])
        feats_b = explanation.get("top_features", [])
        if feats_a and feats_b:
            top_feat_a, top_feat_b = feats_a[0][0], feats_b[0][0]
            fa_name = friendly_feature_name(top_feat_a)[0]
            fb_name = friendly_feature_name(top_feat_b)[0]
            if top_feat_a == top_feat_b:
                st.markdown(f"- Same leading signal on both days: **{fa_name}**.")
            else:
                st.markdown(f"- Leading signal changed from **{fa_name}** to **{fb_name}**.")

    # ---------------- Summary ----------------
    st.subheader("Quick summary for this day")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original forecast (avg mm)", f"{raw_nwp_grid.mean():.1f}",
              help="Average rainfall predicted by the raw model across the whole region.")
    m2.metric("AI-corrected (avg mm)", f"{unet_out['quantile_maps']['q50'].mean():.1f}",
              delta=f"{unet_out['quantile_maps']['q50'].mean() - raw_nwp_grid.mean():.1f} vs original",
              help="Average rainfall after the AI's correction.")
    m3.metric("What actually happened (avg mm)", f"{truth_grid.mean():.1f}",
              help="Average observed rainfall across the whole region.")
    err_raw = abs(raw_nwp_grid.mean() - truth_grid.mean())
    err_corr = abs(unet_out['quantile_maps']['q50'].mean() - truth_grid.mean())
    m4.metric("Error reduced by", f"{(1 - err_corr / (err_raw + 1e-6)) * 100:.0f}%",
              help="How much closer the AI-corrected forecast is to reality, compared to the original.")


if __name__ == "__main__":
    main()
