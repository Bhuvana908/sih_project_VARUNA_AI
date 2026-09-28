"""
End-to-end training pipeline for the Regime-Aware AI Post-Processing system.

Run:
    python train.py

This will:
  1. Generate (or load) the dataset
  2. Train the VAE, fit the GMM -> get regime probabilities ("Weather Mood Ring")
  3. Train the FiLM-conditioned U-Net bias corrector using those probabilities
  4. Train the SHAP surrogate explainer
  5. Evaluate against the raw NWP forecast using RMSE/ETS/CSI/POD/FAR/FSS
  6. Save everything needed by predict.py and the Streamlit dashboard to models/ and outputs/
"""
import os
import json
import yaml
import numpy as np
import torch

from src import data_generation as dg
from src import regime_model as rm
from src import bias_correction as bc
from src import explainability as ex
from src import verification as ver
from src import regime_dynamics as rd

CFG_PATH = "config.yaml"


def load_config():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    np.random.seed(cfg["data"]["seed"])
    torch.manual_seed(cfg["data"]["seed"])

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["outputs_dir"], exist_ok=True)
    os.makedirs(cfg["data"]["sample_dir"], exist_ok=True)

    grid = cfg["domain"]["grid_size"]
    n_regimes = cfg["domain"]["n_regimes"]

    # ---------- 1. Data ----------
    sample_path = os.path.join(cfg["data"]["sample_dir"], "synthetic_dataset.npz")
    subdaily_cfg = cfg.get("subdaily", {})
    n_subdaily_steps = subdaily_cfg.get("steps_per_day", 8)
    subdaily_wobble_std = subdaily_cfg.get("wobble_std", 0.06)

    if os.path.exists(sample_path):
        npz = np.load(sample_path)
        if "subdaily_predictors" in npz.files:
            print(f"[Data] Loading cached dataset from {sample_path}")
            ds = {k: npz[k] for k in npz.files}
        else:
            print(f"[Data] Cached dataset at {sample_path} predates sub-daily support — regenerating.")
            ds = dg.generate_dataset(
                n_seasons=cfg["data"]["n_seasons"],
                season_length=cfg["data"]["season_length_days"],
                grid=grid,
                seed=cfg["data"]["seed"],
                n_subdaily_steps=n_subdaily_steps,
                subdaily_wobble_std=subdaily_wobble_std,
            )
            np.savez_compressed(sample_path, predictors=ds["predictors"], raw_nwp=ds["raw_nwp"],
                                 truth=ds["truth"], regime=ds["regime"], season_id=ds["season_id"],
                                 subdaily_predictors=ds["subdaily_predictors"],
                                 subdaily_regime=ds["subdaily_regime"],
                                 subdaily_hours=ds["subdaily_hours"])
    else:
        print("[Data] Generating synthetic dataset "
              "(swap this for real ERA5 / NCMRWF NWP / IMD gridded rainfall — see README)")
        ds = dg.generate_dataset(
            n_seasons=cfg["data"]["n_seasons"],
            season_length=cfg["data"]["season_length_days"],
            grid=grid,
            seed=cfg["data"]["seed"],
            n_subdaily_steps=n_subdaily_steps,
            subdaily_wobble_std=subdaily_wobble_std,
        )
        np.savez_compressed(sample_path, predictors=ds["predictors"], raw_nwp=ds["raw_nwp"],
                             truth=ds["truth"], regime=ds["regime"], season_id=ds["season_id"],
                             subdaily_predictors=ds["subdaily_predictors"],
                             subdaily_regime=ds["subdaily_regime"], subdaily_hours=ds["subdaily_hours"])

    n_days = ds["predictors"].shape[0]
    test_seasons = cfg["data"]["test_seasons"]
    max_season = ds["season_id"].max()
    test_mask = ds["season_id"] > (max_season - test_seasons)
    train_mask = ~test_mask
    print(f"[Data] {train_mask.sum()} training days, {test_mask.sum()} test days")

    # ---------- 2. Regime identification (VAE + GMM) ----------
    print("[Regime] Normalizing predictors and training VAE ...")
    # Pool each day's own predictor snapshot with its sub-daily snapshots so the
    # regime classifier sees genuine within-day atmospheric variability, not
    # just one sample per day. Sub-daily samples inherit their day's train/test
    # split (a whole day, and all its sub-daily steps, stay on one side).
    n_sub = ds["subdaily_predictors"].shape[1]
    subdaily_flat = ds["subdaily_predictors"].reshape(-1, *ds["subdaily_predictors"].shape[2:])
    subdaily_train_mask = np.repeat(train_mask, n_sub)

    pooled_for_norm = np.concatenate([ds["predictors"][train_mask], subdaily_flat[subdaily_train_mask]], axis=0)
    pmean = pooled_for_norm.mean(axis=(0, 2, 3), keepdims=True)
    pstd = pooled_for_norm.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    predictors_norm_daily = (ds["predictors"] - pmean) / pstd
    subdaily_norm_flat = (subdaily_flat - pmean) / pstd

    vae_train_x = np.concatenate([predictors_norm_daily[train_mask], subdaily_norm_flat[subdaily_train_mask]], axis=0)
    vae = rm.train_vae(
        vae_train_x,
        latent_dim=cfg["vae"]["latent_dim"],
        epochs=cfg["vae"]["epochs"],
        batch_size=cfg["vae"]["batch_size"],
        lr=cfg["vae"]["lr"],
        beta=cfg["vae"]["beta"],
    )
    latents_all = rm.get_latents(vae, predictors_norm_daily)
    latents_subdaily_flat = rm.get_latents(vae, subdaily_norm_flat)

    print("[Regime] Fitting GMM on latent space ...")
    latents_train_pooled = np.concatenate([latents_all[train_mask], latents_subdaily_flat[subdaily_train_mask]], axis=0)
    gmm = rm.fit_gmm(latents_train_pooled, n_components=cfg["gmm"]["n_components"],
                      covariance_type=cfg["gmm"]["covariance_type"], seed=cfg["data"]["seed"])
    regime_probs_all = rm.regime_probabilities(gmm, latents_all)
    regime_probs_subdaily_flat = rm.regime_probabilities(gmm, latents_subdaily_flat)

    cluster_to_name_idx = rm.align_gmm_clusters_to_names(gmm, latents_all[train_mask],
                                                           ds["regime"][train_mask], n_regimes)
    print(f"[Regime] Cluster->regime alignment (for readable reports only): {cluster_to_name_idx}")
    # reorder probability columns to match REGIME_NAMES ordering for readability
    reorder = [k for k, v in sorted(cluster_to_name_idx.items(), key=lambda kv: kv[1])]
    regime_probs_all_aligned = regime_probs_all[:, reorder]
    regime_probs_subdaily_aligned = regime_probs_subdaily_flat[:, reorder].reshape(
        ds["subdaily_predictors"].shape[0], n_sub, n_regimes)

    # ---------- 2b. Regime dynamics: persistence, switch forecasting, novelty ----------
    print("[Regime dynamics] Fitting day-to-day transition matrix and novelty threshold ...")
    hard_cluster_train = gmm.predict(latents_all[train_mask])
    hard_named_train = np.array([cluster_to_name_idx[int(c)] for c in hard_cluster_train])
    transition_matrix = rd.fit_transition_matrix_per_season(
        hard_named_train, cfg["data"]["season_length_days"], n_regimes=n_regimes,
    )
    novelty_threshold = rd.fit_novelty_threshold(gmm, latents_train_pooled, percentile=1.0)

    # ---------- 3. Bias correction (FiLM U-Net) ----------
    print("[U-Net] Training FiLM-conditioned bias correction model ...")
    quantiles = tuple(cfg["uncertainty"]["quantiles"])
    unet = bc.FiLMUNet(in_ch=1, cond_dim=n_regimes, base=cfg["unet"]["base_channels"],
                        dropout_p=cfg["unet"]["dropout_p"], quantiles=quantiles)
    unet = bc.train_unet(
        unet,
        ds["raw_nwp"][train_mask], regime_probs_all_aligned[train_mask], ds["truth"][train_mask],
        epochs=cfg["unet"]["epochs"], batch_size=cfg["unet"]["batch_size"], lr=cfg["unet"]["lr"],
        quantiles=quantiles,
        heavy_threshold=cfg["unet"]["heavy_rain_weight_threshold_mm"],
        heavy_weight=cfg["unet"]["heavy_rain_weight_factor"],
    )

    # ---------- 4. Explainability surrogate ----------
    print("[XAI] Training SHAP surrogate on interpretable summary features ...")
    X_all, feature_names = ex.build_feature_matrix(ds["predictors"], list(ds.get(
        "predictor_names", dg.PREDICTOR_CHANNELS)))
    surrogate = ex.train_surrogate(X_all[train_mask], regime_probs_all_aligned[train_mask])

    # ---------- 5. Evaluate on held-out test seasons ----------
    print("[Eval] Generating corrected forecasts + evaluating on test set ...")
    unet.eval()
    test_idx = np.where(test_mask)[0]
    corrected_test = np.zeros((len(test_idx), grid, grid), dtype=np.float32)
    x_test = torch.tensor(ds["raw_nwp"][test_idx], dtype=torch.float32).unsqueeze(1)
    c_test = torch.tensor(regime_probs_all_aligned[test_idx], dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(test_idx), 32):
            xb = x_test[i:i + 32]
            cb = c_test[i:i + 32]
            out = unet(xb, cb)
            median_idx = out.shape[1] // 2
            corrected_test[i:i + 32] = out[:, median_idx].numpy()

    thresholds = cfg["verification"]["thresholds_mm"]
    neighborhood = cfg["verification"]["fss_neighborhood"]
    report_raw = ver.full_report(ds["raw_nwp"][test_idx], ds["truth"][test_idx], thresholds, neighborhood)
    report_corrected = ver.full_report(corrected_test, ds["truth"][test_idx], thresholds, neighborhood)

    print("\n=== VERIFICATION REPORT (test seasons) ===")
    print(f"{'Metric':<12}{'Raw NWP':>12}{'AI-Corrected':>16}")
    for k in report_raw:
        print(f"{k:<12}{report_raw[k]:>12.3f}{report_corrected[k]:>16.3f}")

    # ---------- 6. Save everything ----------
    models_dir = cfg["paths"]["models_dir"]
    torch.save(vae.state_dict(), os.path.join(models_dir, "vae.pt"))
    torch.save(unet.state_dict(), os.path.join(models_dir, "unet.pt"))
    np.savez(os.path.join(models_dir, "vae_norm.npz"), mean=pmean, std=pstd)
    import pickle
    with open(os.path.join(models_dir, "gmm.pkl"), "wb") as f:
        pickle.dump({"gmm": gmm, "cluster_to_name_idx": cluster_to_name_idx}, f)
    with open(os.path.join(models_dir, "surrogate.pkl"), "wb") as f:
        pickle.dump({"model": surrogate, "feature_names": feature_names}, f)
    with open(os.path.join(models_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Regime dynamics artifacts (transition matrix, novelty threshold, and the
    # training latents/labels analog-day retrieval searches over)
    np.savez(
        os.path.join(models_dir, "regime_dynamics.npz"),
        transition_matrix=transition_matrix,
        novelty_threshold=novelty_threshold,
        train_latents=latents_all[train_mask],
        train_hard_regime_named=hard_named_train,
    )

    np.savez_compressed(
        os.path.join(cfg["paths"]["outputs_dir"], "test_results.npz"),
        test_idx=test_idx, raw_nwp=ds["raw_nwp"][test_idx], truth=ds["truth"][test_idx],
        corrected=corrected_test, regime_probs=regime_probs_all_aligned[test_idx],
        predictors=ds["predictors"][test_idx],
        subdaily_regime_probs=regime_probs_subdaily_aligned[test_idx],
        subdaily_regime_true=ds["subdaily_regime"][test_idx],
        subdaily_hours=ds["subdaily_hours"],
    )
    with open(os.path.join(cfg["paths"]["outputs_dir"], "verification_report.json"), "w") as f:
        json.dump({"raw_nwp": report_raw, "ai_corrected": report_corrected}, f, indent=2)

    print(f"\n[Done] Models saved to {models_dir}/, outputs saved to {cfg['paths']['outputs_dir']}/")
    print("Run the dashboard with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
