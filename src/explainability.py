"""
Step 4: The "Detective" — Explainable AI.

The full pipeline (conv-VAE -> GMM -> FiLM U-Net) is not directly SHAP-friendly
(high-dimensional image input, non-differentiable GMM step). Instead we train a
lightweight, interpretable SURROGATE model that maps physically meaningful
SUMMARY indices (domain-mean and quadrant-mean winds/moisture/pressure/CAPE -
exactly the kind of variables a forecaster reasons about) to the same regime
probability vector produced by the real VAE+GMM pipeline. SHAP explains this
surrogate, which is standard practice for explaining complex geophysical AI
pipelines and gives forecaster-readable output like the brief's own example:
"reduced rainfall because 850 hPa winds were easterly and CAPE was low".
"""
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor


def summary_features(predictor_grid, predictor_names):
    """predictor_grid: (C, H, W) -> dict of interpretable scalar features:
    domain mean + 4 quadrant means per channel."""
    c, h, w = predictor_grid.shape
    hh, hw = h // 2, w // 2
    quads = {
        "NW": predictor_grid[:, :hh, :hw],
        "NE": predictor_grid[:, :hh, hw:],
        "SW": predictor_grid[:, hh:, :hw],
        "SE": predictor_grid[:, hh:, hw:],
    }
    feats = {}
    for ci, name in enumerate(predictor_names):
        feats[f"{name}_domain_mean"] = predictor_grid[ci].mean()
        for qname, qgrid in quads.items():
            feats[f"{name}_{qname}"] = qgrid[ci].mean()
    return feats


def build_feature_matrix(predictors, predictor_names):
    rows = [summary_features(predictors[i], predictor_names) for i in range(predictors.shape[0])]
    feature_names = list(rows[0].keys())
    X = np.array([[r[k] for k in feature_names] for r in rows])
    return X, feature_names


def train_surrogate(X, regime_probs, seed=42):
    """Multi-output RandomForest trained to reproduce the real pipeline's
    regime-probability output from interpretable summary features."""
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed)
    model.fit(X, regime_probs)
    return model


def explain_day(surrogate, X_row, feature_names, regime_names, top_k=5):
    """Return top-k feature attributions (SHAP values) for the predicted
    top regime on a single day, plus a human-readable sentence."""
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X_row.reshape(1, -1))  # list per output OR array

    probs = surrogate.predict(X_row.reshape(1, -1))[0]
    top_regime_idx = int(np.argmax(probs))

    # shap_values shape handling across shap versions / multi-output sklearn models:
    #   - list of length n_outputs, each (n_samples, n_features)          [older shap]
    #   - ndarray (n_outputs, n_samples, n_features)                       [some versions]
    #   - ndarray (n_samples, n_features, n_outputs)                       [newer shap, observed here]
    sv = shap_values
    if isinstance(sv, list):
        vals = np.asarray(sv[top_regime_idx])[0]
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3:
            if sv.shape[0] == len(regime_names):          # (n_outputs, n_samples, n_features)
                vals = sv[top_regime_idx, 0, :]
            elif sv.shape[-1] == len(regime_names):        # (n_samples, n_features, n_outputs)
                vals = sv[0, :, top_regime_idx]
            else:
                vals = sv[0].reshape(-1)
        else:
            vals = sv[0]

    order = np.argsort(-np.abs(vals))[:top_k]
    top_features = [(feature_names[i], float(vals[i]), float(X_row[i])) for i in order]

    parts = []
    for fname, val, raw in top_features:
        direction = "increased" if val > 0 else "decreased"
        parts.append(f"{fname}={raw:.1f} {direction} confidence in '{regime_names[top_regime_idx]}' "
                      f"(impact {val:+.3f})")
    sentence = f"Predicted regime: {regime_names[top_regime_idx]} ({probs[top_regime_idx]*100:.0f}% confidence). " \
               + " ; ".join(parts)

    return {
        "top_regime": regime_names[top_regime_idx],
        "top_regime_confidence": float(probs[top_regime_idx]),
        "top_features": top_features,
        "explanation_text": sentence,
    }
