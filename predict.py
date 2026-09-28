"""
Load trained models and run the full pipeline (regime ID -> bias correction ->
uncertainty -> explanation) for a chosen day. Used by the Streamlit dashboard,
and runnable standalone for a quick sanity check:

    python predict.py --day 10
"""
import argparse
import pickle
import numpy as np
import torch
import yaml

from src import regime_model as rm
from src import bias_correction as bc
from src import uncertainty as unc
from src import explainability as ex
from src import data_generation as dg
from src import regime_dynamics as rd


def load_all(cfg):
    models_dir = cfg["paths"]["models_dir"]
    grid = cfg["domain"]["grid_size"]
    n_regimes = cfg["domain"]["n_regimes"]

    norm = np.load(f"{models_dir}/vae_norm.npz")
    pmean, pstd = norm["mean"], norm["std"]

    vae = rm.ConvVAE(len(dg.PREDICTOR_CHANNELS), grid, cfg["vae"]["latent_dim"])
    vae.load_state_dict(torch.load(f"{models_dir}/vae.pt", map_location="cpu"))
    vae.eval()

    with open(f"{models_dir}/gmm.pkl", "rb") as f:
        gmm_data = pickle.load(f)
    gmm, cluster_to_name_idx = gmm_data["gmm"], gmm_data["cluster_to_name_idx"]
    reorder = [k for k, v in sorted(cluster_to_name_idx.items(), key=lambda kv: kv[1])]

    quantiles = tuple(cfg["uncertainty"]["quantiles"])
    unet = bc.FiLMUNet(in_ch=1, cond_dim=n_regimes, base=cfg["unet"]["base_channels"],
                        dropout_p=cfg["unet"]["dropout_p"], quantiles=quantiles)
    unet.load_state_dict(torch.load(f"{models_dir}/unet.pt", map_location="cpu"))
    unet.eval()

    with open(f"{models_dir}/surrogate.pkl", "rb") as f:
        surr_data = pickle.load(f)

    rdyn = np.load(f"{models_dir}/regime_dynamics.npz")

    return {
        "vae": vae, "pmean": pmean, "pstd": pstd, "gmm": gmm, "reorder": reorder,
        "unet": unet, "quantiles": quantiles,
        "surrogate": surr_data["model"], "feature_names": surr_data["feature_names"],
        "transition_matrix": rdyn["transition_matrix"],
        "novelty_threshold": float(rdyn["novelty_threshold"]),
        "train_latents": rdyn["train_latents"],
        "train_hard_regime_named": rdyn["train_hard_regime_named"],
    }


def run_for_day(assets, predictor_grid, raw_nwp_grid, cfg):
    """predictor_grid: (C,H,W) raw (unnormalized). raw_nwp_grid: (H,W)."""
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

    # Regime dynamics: persistence-aware switch forecast + novelty flag + analogs
    trans = assets["transition_matrix"]
    switch_in_2_days = rd.probability_of_regime_switch(regime_probs, trans, horizon_days=2)
    expected_switch_day = rd.days_until_likely_switch(regime_probs, trans)
    is_novel, novelty_score = rd.is_novel_regime(assets["gmm"], latent, assets["novelty_threshold"])
    analogs = rd.find_analog_days(
        latent, assets["train_latents"], assets["train_hard_regime_named"],
        cfg["domain"]["regime_names"], k=3,
    )

    return {
        "regime_probs": dict(zip(cfg["domain"]["regime_names"], regime_probs.tolist())),
        "corrected_median": unet_out["quantile_maps"]["q50"],
        "uncertainty": unet_out,
        "explanation": explanation,
        "switch_in_2_days_prob": switch_in_2_days,
        "expected_switch_day": expected_switch_day,
        "is_novel_regime": is_novel,
        "novelty_score": novelty_score,
        "analog_days": analogs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, default=0, help="index into cached test set")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    assets = load_all(cfg)
    npz = np.load("outputs/test_results.npz")
    predictors = npz["predictors"]
    raw_nwp = npz["raw_nwp"]
    truth = npz["truth"]

    d = args.day
    result = run_for_day(assets, predictors[d], raw_nwp[d], cfg)

    print("Regime probabilities:")
    for k, v in result["regime_probs"].items():
        print(f"  {k:<22} {v*100:5.1f}%")
    print("\nExplanation:")
    print(" ", result["explanation"]["explanation_text"])
    print("\nRaw NWP mean rainfall:      ", raw_nwp[d].mean().round(2))
    print("AI-corrected mean rainfall: ", result["corrected_median"].mean().round(2))
    print("Truth mean rainfall:        ", truth[d].mean().round(2))
    print("P(rain >= threshold) domain-mean:", result["uncertainty"]["exceedance_probability"].mean().round(3))

    print("\nRegime outlook:")
    print(f"  Chance current regime ends within 2 days: {result['switch_in_2_days_prob']*100:.0f}%")
    if result["expected_switch_day"] is not None:
        print(f"  Expect a regime change in ~{result['expected_switch_day']} day(s)")
    else:
        print("  No high-confidence regime change expected in the next 7 days")
    if result["is_novel_regime"]:
        print(f"  Warning: today's atmospheric state doesn't closely match any known regime "
              f"(novelty score {result['novelty_score']:.2f})")
    print("  Closest historical analog days:")
    for a in result["analog_days"]:
        print(f"    Day {a['day_index']}: {a['regime_name']} "
              f"(persisted {a['days_persisted_after']} more day(s) after)")
