"""
Regime Dynamics — turns the static, day-by-day "Weather Mood Ring" (VAE+GMM)
into a temporally-aware system.

The existing pipeline (src/regime_model.py) treats every day independently:
each day gets a soft regime probability vector from the GMM, with no memory
of what the regime was yesterday. This module adds three things on top of
that, without changing the VAE or GMM themselves:

  1. Temporal smoothing of the daily regime probabilities using a learned
     Markov transition matrix (a lightweight HMM-style filter), so the
     output respects real monsoon persistence instead of jittering day to day.

  2. Regime-switch forecasting: given today's (smoothed) regime distribution,
     project it forward N days using the transition matrix and report the
     probability that the dominant regime changes — e.g. "62% chance Break
     Monsoon ends within 2 days." This is new information the raw GMM output
     does not give at all; NWP forecasts tell you tomorrow's rainfall, not
     whether the current monsoon "mood" is about to flip.

  3. Novel-regime detection: a day whose latent-space log-likelihood under
     every GMM component is far below what training data ever produced is
     flagged as "doesn't match any of the 5 known regimes," rather than
     being silently forced into the closest one.

  4. Analog-day retrieval: for the current latent vector, find the k most
     similar historical days and report how long that regime persisted
     after them. Useful for the dashboard / for forecaster trust — turns an
     abstract probability vector into "this looks like the Break Monsoon
     spell of day 214, which lasted 5 more days."

None of this requires retraining the VAE or GMM. It consumes their existing
outputs (`latents`, `gmm`, `regime_probabilities(...)` from regime_model.py).
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. Transition matrix estimation
# ---------------------------------------------------------------------------

def fit_transition_matrix(hard_regime_sequence, n_regimes, laplace_smoothing=1.0):
    """
    Estimate a day-to-day Markov transition matrix from a historical sequence
    of hard regime labels (e.g. gmm.predict(latents) run on the *training*
    set, in chronological order, with season boundaries either kept or
    dropped — see `season_breaks` below if you want to exclude the
    day-1-of-a-season -> day-1-of-next-season jump).

    Returns an (n_regimes, n_regimes) row-stochastic matrix T where
    T[i, j] = P(tomorrow's regime = j | today's regime = i).

    Laplace smoothing avoids zero-probability transitions for regime pairs
    that never happened to co-occur in the (necessarily limited) training
    record — important for Western Disturbance / Monsoon Depression, which
    are rare regimes with few observed transitions.
    """
    counts = np.full((n_regimes, n_regimes), laplace_smoothing, dtype=np.float64)
    seq = np.asarray(hard_regime_sequence)
    for t in range(len(seq) - 1):
        counts[seq[t], seq[t + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def fit_transition_matrix_per_season(hard_regime_sequence, season_length_days,
                                      n_regimes, laplace_smoothing=1.0):
    """
    Same as fit_transition_matrix, but skips the artificial jump between the
    last day of one synthetic season and the first day of the next (they are
    not really consecutive days). Use this with the synthetic generator's
    `season_length_days` (see config.yaml -> data.season_length_days).
    """
    seq = np.asarray(hard_regime_sequence)
    counts = np.full((n_regimes, n_regimes), laplace_smoothing, dtype=np.float64)
    for t in range(len(seq) - 1):
        if (t + 1) % season_length_days == 0:
            continue  # season boundary, not a real day-to-day transition
        counts[seq[t], seq[t + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


# ---------------------------------------------------------------------------
# 2. Temporal smoothing (Bayesian filtering over the transition matrix)
# ---------------------------------------------------------------------------

def smooth_regime_probs(raw_probs_sequence, transition_matrix):
    """
    Forward-filter a sequence of raw per-day GMM regime probabilities into a
    temporally coherent sequence, using the transition matrix as a prior on
    what tomorrow's regime should look like given today's.

    raw_probs_sequence: (T, n_regimes) array, one row per day, from
        regime_probabilities(gmm, latents) in chronological order.
    transition_matrix: (n_regimes, n_regimes) from fit_transition_matrix*.

    Returns (T, n_regimes) smoothed probabilities. Day 0 is unchanged (no
    prior day to condition on); each subsequent day blends the GMM's raw
    emission with the predicted distribution from the previous smoothed day.
    """
    raw = np.asarray(raw_probs_sequence)
    T, n_regimes = raw.shape
    smoothed = np.zeros_like(raw)
    smoothed[0] = raw[0]
    for t in range(1, T):
        predicted = smoothed[t - 1] @ transition_matrix   # prior for today
        posterior = predicted * raw[t]                    # combine with GMM's own estimate
        posterior_sum = posterior.sum()
        smoothed[t] = posterior / posterior_sum if posterior_sum > 1e-12 else raw[t]
    return smoothed


# ---------------------------------------------------------------------------
# 3. Regime-switch forecasting
# ---------------------------------------------------------------------------

def forecast_regime_distribution(current_probs, transition_matrix, horizon_days):
    """
    Project today's regime distribution forward by `horizon_days` using the
    transition matrix (matrix power). Returns the forecast distribution at
    t + horizon_days.
    """
    T_h = np.linalg.matrix_power(transition_matrix, horizon_days)
    return current_probs @ T_h


def probability_of_regime_switch(current_probs, transition_matrix, horizon_days):
    """
    Probability that the *dominant* regime at t + horizon_days differs from
    today's dominant regime. This is the headline number for a "regime
    switch alert": e.g. probability_of_regime_switch(...) = 0.62 means a
    62% chance the current monsoon "mood" changes within `horizon_days`.
    """
    current_dominant = int(np.argmax(current_probs))
    future_dist = forecast_regime_distribution(current_probs, transition_matrix, horizon_days)
    return 1.0 - float(future_dist[current_dominant])


def days_until_likely_switch(current_probs, transition_matrix, max_horizon=7, switch_prob_threshold=0.5):
    """
    Scans horizons 1..max_horizon and returns the first horizon at which
    probability_of_regime_switch exceeds switch_prob_threshold, or None if it
    never crosses that threshold within max_horizon days. Convenience
    wrapper for a single "expect a change in ~N days" dashboard readout.
    """
    for h in range(1, max_horizon + 1):
        if probability_of_regime_switch(current_probs, transition_matrix, h) >= switch_prob_threshold:
            return h
    return None


# ---------------------------------------------------------------------------
# 4. Novel-regime (out-of-distribution) detection
# ---------------------------------------------------------------------------

def novel_regime_score(gmm, latent_vector):
    """
    Log-likelihood of a single day's latent vector under the fitted GMM
    (gmm.score_samples expects a batch, so this wraps a single sample).
    Lower (more negative) = less like anything seen in training.
    """
    return float(gmm.score_samples(latent_vector.reshape(1, -1))[0])


def fit_novelty_threshold(gmm, training_latents, percentile=1.0):
    """
    Establishes a novelty threshold from the training set itself: the
    `percentile`-th percentile of training log-likelihoods. Days scoring
    below this in deployment are flagged as "doesn't match any known
    regime well" rather than being silently forced into the nearest one.
    Default percentile=1.0 flags roughly the bottom 1% of days seen so far
    as the boundary of "normal."
    """
    scores = gmm.score_samples(training_latents)
    return float(np.percentile(scores, percentile))


def is_novel_regime(gmm, latent_vector, novelty_threshold):
    score = novel_regime_score(gmm, latent_vector)
    return score < novelty_threshold, score


# ---------------------------------------------------------------------------
# 5. Analog-day retrieval
# ---------------------------------------------------------------------------

def find_analog_days(current_latent, historical_latents, historical_hard_regime,
                      regime_names, k=3):
    """
    Nearest neighbours (Euclidean, in VAE latent space) to the current day
    among historical days. For each analog, also reports how many further
    consecutive days that same regime persisted historically — turns the
    abstract regime probability vector into a concrete, checkable precedent
    for forecasters ("closest match: day 214, a Break Monsoon spell that
    then lasted 5 more days").

    historical_latents: (N, latent_dim), chronological order.
    historical_hard_regime: (N,) int array, gmm.predict(historical_latents)
        or the aligned/named regime index for each day, same order.
    """
    dists = np.linalg.norm(historical_latents - current_latent.reshape(1, -1), axis=1)
    nearest_idx = np.argsort(dists)[:k]

    results = []
    for idx in nearest_idx:
        regime = int(historical_hard_regime[idx])
        persistence = 1
        j = idx + 1
        while j < len(historical_hard_regime) and historical_hard_regime[j] == regime:
            persistence += 1
            j += 1
        results.append({
            "day_index": int(idx),
            "distance": float(dists[idx]),
            "regime_name": regime_names[regime] if regime_names else str(regime),
            "days_persisted_after": persistence,
        })
    return results
