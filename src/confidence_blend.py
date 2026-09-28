"""
Confidence-Aware Forecast Handoff.

The pipeline currently always trusts the FiLM U-Net's correction completely,
regardless of how confident the regime classification underneath it was. But
the correction is conditioned on the regime probability vector — if the GMM
itself is unsure what regime today is (a near-even split between two
regimes), or has flagged today as not matching any known regime at all (see
src/regime_dynamics.py's novelty detection), the correction it produces is
built on a shaky foundation and shouldn't be trusted as fully as a day with
a clean, confident regime call.

This module computes a single confidence score per day and uses it to blend
the AI-corrected forecast back towards the raw NWP forecast when that
confidence is low — instead of an all-or-nothing choice between "always use
the AI" and "always use the raw forecast." This is a real, if simple,
mixture-of-experts-style gating mechanism: the model's own uncertainty about
its regime call directly controls how much it is allowed to change the
forecast.
"""
import numpy as np


def regime_confidence(regime_probs, n_regimes):
    """
    Confidence in the regime call, from 0 (no better than random guessing
    among n_regimes) to 1 (fully certain). Uses the top regime's probability,
    rescaled so that the "uninformative" baseline (1/n_regimes, i.e. every
    regime equally likely) maps to 0 rather than to some arbitrary positive
    number.
    """
    top_p = float(np.max(regime_probs))
    baseline = 1.0 / n_regimes
    if baseline >= 1.0:
        return 1.0
    conf = (top_p - baseline) / (1.0 - baseline)
    return float(np.clip(conf, 0.0, 1.0))


def effective_confidence(regime_probs, n_regimes, is_novel_regime, novelty_ceiling=0.3):
    """
    Combines regime_confidence with the novelty flag from regime_dynamics.py.
    A novel-regime day caps confidence at `novelty_ceiling` regardless of how
    peaked the (untrustworthy) regime probability vector looks — a confident
    -looking probability vector on a day the GMM has never really seen
    before is not actually trustworthy.
    """
    conf = regime_confidence(regime_probs, n_regimes)
    if is_novel_regime:
        conf = min(conf, novelty_ceiling)
    return conf


def blend_forecast(raw_nwp_grid, corrected_grid, confidence):
    """
    Weighted cell-wise blend between the raw NWP forecast and the AI
    -corrected forecast: confidence=1 returns the AI correction unchanged,
    confidence=0 returns the raw forecast unchanged, and values in between
    interpolate linearly.
    """
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return confidence * corrected_grid + (1.0 - confidence) * raw_nwp_grid
