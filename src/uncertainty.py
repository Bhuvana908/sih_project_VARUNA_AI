"""
Step 3: The "Probability Map" — Heavy Rainfall Uncertainty.

Combines:
  1. Quantile regression heads already trained inside the U-Net (10th/50th/90th
     percentile of corrected rainfall).
  2. Monte Carlo Dropout: running the network many times with dropout active to
     build an empirical ensemble, from which we derive exceedance probabilities
     such as "85% chance of >100mm".
"""
import numpy as np
import torch
from .bias_correction import mc_dropout_predict


def predict_with_uncertainty(model, raw_nwp_single, regime_prob_single, quantiles,
                              n_mc_passes=50, heavy_threshold=100.0, device="cpu"):
    model.to(device)
    model.eval()
    x = torch.tensor(raw_nwp_single, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    cond = torch.tensor(regime_prob_single, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x, cond)[0].cpu().numpy()  # (n_q, H, W)
    quantile_maps = {f"q{int(q*100)}": out[i] for i, q in enumerate(quantiles)}

    mc_samples = mc_dropout_predict(model, raw_nwp_single, regime_prob_single,
                                     n_passes=n_mc_passes, device=device)
    stack = np.stack(mc_samples, axis=0)          # (n_passes, H, W)
    mean_map = stack.mean(axis=0)
    std_map = stack.std(axis=0)
    exceed_prob = (stack >= heavy_threshold).mean(axis=0)   # P(rain >= threshold)

    return {
        "quantile_maps": quantile_maps,
        "mc_mean": mean_map,
        "mc_std": std_map,
        "exceedance_probability": exceed_prob,
        "heavy_threshold": heavy_threshold,
    }
