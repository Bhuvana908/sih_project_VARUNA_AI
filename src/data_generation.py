"""
Synthetic data generator standing in for:
  - ERA5 reanalysis predictors (u850, v850, q850, mslp anomaly, CAPE)
  - Raw NWP rainfall forecast (biased)
  - "Truth" / IMD-gridded observed rainfall

WHY SYNTHETIC DATA:
Real ERA5 data requires a free CDS API key + multi-GB downloads, and real
NWP/IMD gridded rainfall requires NCMRWF/IMD data access that is not available
in this environment. This generator produces physically-plausible, internally
consistent data (each of the 5 monsoon regimes has a distinct spatial rainfall
signature AND a distinct NWP bias signature AND distinct atmospheric predictor
statistics) so the full pipeline can be trained and evaluated end-to-end.

See README.md -> "Using real data" for how to swap this out for actual
ERA5 / NCMRWF / IMD files with the same array shapes.
"""
import numpy as np

REGIME_NAMES = [
    "Active Monsoon",
    "Break Monsoon",
    "Monsoon Depression",
    "Orographic Rainfall",
    "Western Disturbance",
]
N_REGIMES = len(REGIME_NAMES)
PREDICTOR_CHANNELS = ["u850", "v850", "q850", "mslp_anom", "cape"]
N_PREDICTORS = len(PREDICTOR_CHANNELS)


def _gaussian_blob(grid, cx, cy, sigma, amp):
    y, x = np.mgrid[0:grid, 0:grid]
    return amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))


def _smooth_field(grid, rng, n_modes=4, amp=1.0):
    """Low-frequency random field via a small sum of 2D sinusoids (cheap substitute
    for spatially correlated noise, avoids extra scipy dependency)."""
    y, x = np.mgrid[0:grid, 0:grid] / grid
    field = np.zeros((grid, grid))
    for _ in range(n_modes):
        kx, ky = rng.uniform(0.5, 2.5, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        field += np.sin(2 * np.pi * (kx * x + ky * y) + phase)
    field -= field.mean()
    field /= (field.std() + 1e-8)
    return amp * field


def true_rainfall_pattern(regime_idx, grid, rng):
    """Ground-truth ('observed'/IMD-like) rainfall field for a given regime."""
    field = np.zeros((grid, grid))
    if regime_idx == 0:  # Active Monsoon: widespread heavy rain everywhere
        field += 40 + 10 * _smooth_field(grid, rng, amp=1.0)
        field += _gaussian_blob(grid, grid * 0.55, grid * 0.5, grid * 0.7, 25)
    elif regime_idx == 1:  # Break Monsoon: dry interior, light rain only near coasts
        field += 2 + 1.5 * np.abs(_smooth_field(grid, rng, amp=1.0))
        field += _gaussian_blob(grid, grid * 0.12, grid * 0.5, grid * 0.22, 14)   # west coast
        field += _gaussian_blob(grid, grid * 0.88, grid * 0.78, grid * 0.18, 9)   # northeast
    elif regime_idx == 2:  # Monsoon Depression: intense localized vortex, moves each day
        cx = rng.uniform(grid * 0.3, grid * 0.75)
        cy = rng.uniform(grid * 0.35, grid * 0.75)
        field += 8 + 4 * np.abs(_smooth_field(grid, rng, amp=1.0))
        field += _gaussian_blob(grid, cx, cy, grid * 0.16, 120)
    elif regime_idx == 3:  # Orographic: strong band along west edge (Western Ghats proxy)
        field += 6 + 2 * np.abs(_smooth_field(grid, rng, amp=1.0))
        field += _gaussian_blob(grid, grid * 0.1, grid * 0.5, grid * 0.4, 65)
    else:  # Western Disturbance: enhanced rain over north / northwest
        field += 4 + 2 * np.abs(_smooth_field(grid, rng, amp=1.0))
        field += _gaussian_blob(grid, grid * 0.28, grid * 0.1, grid * 0.32, 50)
    return np.clip(field, 0, None)


def raw_nwp_forecast(true_field, regime_idx, grid, rng):
    """Simulate a biased raw NWP forecast derived from the truth field, with a
    distinct, physically-motivated bias signature per regime."""
    field = true_field.copy()

    def box_blur(f, k=3):
        pad = k // 2
        fp = np.pad(f, pad, mode="edge")
        out = np.zeros_like(f)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                out[i, j] = fp[i:i + k, j:j + k].mean()
        return out

    if regime_idx == 0:  # Active Monsoon: NWP over-forecasts intensity, smooths it out
        field = box_blur(field, 3) * 1.30
    elif regime_idx == 1:  # Break Monsoon: NWP adds spurious light rain (false alarms)
        field = field * 0.9 + _gaussian_blob(grid, rng.uniform(0, grid), rng.uniform(0, grid), grid * 0.25, 8)
    elif regime_idx == 2:  # Monsoon Depression: NWP under-forecasts peak & mislocates centre
        shift = rng.integers(-3, 4, size=2)
        field = np.roll(field, shift, axis=(0, 1)) * 0.55
        field = box_blur(field, 3)
    elif regime_idx == 3:  # Orographic: NWP badly under-resolves terrain enhancement
        field = box_blur(field, 5) * 0.45
    else:  # Western Disturbance: NWP over-spreads the system
        field = box_blur(field, 5) * 1.25

    field += np.abs(rng.normal(0, 3.0, size=field.shape))
    return np.clip(field, 0, None)


REGIME_SIGNATURES = {
    0: (8.0, 6.0, 16.0, -4.0, 1800),   # Active: strong westerlies, high moisture, low pressure, high CAPE
    1: (2.0, -1.0, 9.0, 3.0, 500),     # Break: weak/reversed flow, dry, high pressure, low CAPE
    2: (5.0, 9.0, 18.0, -8.0, 2200),   # Depression: strong cyclonic inflow, very moist, deep low, high CAPE
    3: (10.0, 3.0, 14.0, -2.0, 1200),  # Orographic: strong onshore flow against Ghats
    4: (-3.0, 4.0, 7.0, 1.0, 400),     # Western Disturbance: westerly upper trough, drier, cooler
}


def predictor_fields_from_weights(weights, grid, rng, noise_scale=0.55):
    """Synthetic multi-channel atmospheric predictor grid (ERA5-like), built as a
    *blend* of regime signatures according to `weights` (a probability-like vector
    over regimes). Using a continuous blend + realistic noise (rather than a hard
    regime -> field mapping) is what makes the day-to-day atmospheric state trace
    out smooth transitions in latent space, so the downstream GMM produces genuine
    fractional regime probabilities near transitions instead of a hard 0/1 switch."""
    n_regimes = len(REGIME_SIGNATURES)
    base = np.zeros(len(REGIME_SIGNATURES[0]))
    for r in range(n_regimes):
        base += weights[r] * np.array(REGIME_SIGNATURES[r])

    channels = []
    for b in base:
        amp = max(abs(b) * 0.35, 1.0)
        ch = (b
              + amp * _smooth_field(grid, rng, n_modes=3, amp=1.0)
              + rng.normal(0, amp * noise_scale, size=(grid, grid)))
        channels.append(ch)
    return np.stack(channels, axis=0)  # (N_PREDICTORS, grid, grid)


def predictor_fields(regime_idx, grid, rng):
    """Convenience wrapper: pure (non-blended) regime signature."""
    w = np.zeros(len(REGIME_SIGNATURES))
    w[regime_idx] = 1.0
    return predictor_fields_from_weights(w, grid, rng)


def smooth_regime_weights(seq, n_regimes, window=5):
    """Turn a discrete regime sequence into smoothly-blended probability-like
    weights via a moving average over `window` days, so transitions between
    regimes are gradual rather than instantaneous (physically realistic, and
    necessary for the GMM to ever output soft/fractional probabilities)."""
    n = len(seq)
    onehot = np.zeros((n, n_regimes))
    onehot[np.arange(n), seq] = 1.0
    pad = window // 2
    padded = np.pad(onehot, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.zeros_like(onehot)
    for i in range(n):
        smoothed[i] = padded[i:i + window].mean(axis=0)
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return smoothed


def _markov_regime_sequence(n_days, n_regimes, rng, persistence=0.85, regime_prior=None):
    """Generate a realistic regime sequence with day-to-day persistence
    (monsoon regimes typically last several days to weeks).

    regime_prior: optional array of length n_regimes giving the relative
    long-run frequency of each regime (does not need to sum to 1). Used so
    that a genuinely rare-but-real state (e.g. Western Disturbance influence
    during the monsoon) occurs occasionally rather than as often as the
    monsoon's own core regimes."""
    if regime_prior is None:
        regime_prior = np.ones(n_regimes)
    regime_prior = np.asarray(regime_prior, dtype=float)

    start_probs = regime_prior / regime_prior.sum()
    seq = [rng.choice(n_regimes, p=start_probs)]
    for _ in range(1, n_days):
        current = seq[-1]
        off_mass = regime_prior.copy()
        off_mass[current] = 0.0
        off_mass = off_mass / off_mass.sum() * (1 - persistence)
        probs = off_mass
        probs[current] = persistence
        seq.append(rng.choice(n_regimes, p=probs))
    return np.array(seq)


def _subdaily_weight_trajectory(w_prev, w_today, w_next, n_steps, rng, wobble_std=0.06):
    """Build a genuine sub-daily (hourly-resolution) regime-weight trajectory
    for one day, instead of a single fixed daily weight vector.

    Each day's weight vector (w_prev/w_today/w_next) is treated as the state
    at local noon of the previous/current/next day. We piecewise-linearly
    interpolate between those three anchors to get the day's smooth
    day-to-day evolution, then add a small independent Gaussian wobble at
    each sub-daily step (representing sub-daily/synoptic-scale variability
    that isn't captured by day-to-day persistence alone) before renormalizing
    back to a valid probability-like vector. This is what lets the same day
    genuinely drift from one regime's neighbourhood to another's over the
    course of 24h, rather than just recoloring one fixed field.
    """
    n_regimes = len(w_today)
    hours = np.linspace(0, 24, n_steps, endpoint=False)
    traj = np.zeros((n_steps, n_regimes))
    for i, h in enumerate(hours):
        if h <= 12:
            frac = (h + 12) / 24.0
            base = w_prev + frac * (w_today - w_prev)
        else:
            frac = (h - 12) / 24.0
            base = w_today + frac * (w_next - w_today)
        wobble = rng.normal(0, wobble_std, size=n_regimes)
        w = np.clip(base + wobble, 1e-4, None)
        traj[i] = w / w.sum()
    return traj, hours


def generate_dataset(n_seasons=8, season_length=122, grid=24, seed=42, regime_prior=None,
                      n_subdaily_steps=8, subdaily_wobble_std=0.06):
    """Returns a dict of numpy arrays, one row per simulated day:
        predictors : (N, N_PREDICTORS, grid, grid)
        raw_nwp    : (N, grid, grid)
        truth      : (N, grid, grid)
        regime     : (N,)  integer ground-truth regime (for validation only;
                            the pipeline itself never uses this as a training label)
        season_id  : (N,)

    Plus genuine (not interpolated-at-inference-time) sub-daily atmospheric
    state, used to study/display within-day regime transitions:
        subdaily_predictors : (N, n_subdaily_steps, N_PREDICTORS, grid, grid)
        subdaily_regime     : (N, n_subdaily_steps)  discrete label per step,
                               validation-only, same caveat as `regime`
        subdaily_hours      : (n_subdaily_steps,)  hour-of-day for each step

    Daily rainfall (raw_nwp/truth) stays at daily resolution, matching real
    IMD/NCMRWF gridded products, which are day-resolution; only the
    atmospheric predictors (which real ERA5 reanalysis provides at 6-hourly
    or finer resolution) are given genuine sub-daily variation.

    regime_prior: relative long-run frequency of each regime. Defaults to a
    monsoon-realistic mix where Active/Break/Orographic are common, Monsoon
    Depression is an occasional high-impact event, and Western Disturbance
    influence is rare during the core monsoon season (it is predominantly a
    winter phenomenon; the problem statement lists it as a state the system
    should still be able to recognize on the rare days it does occur)."""
    if regime_prior is None:
        regime_prior = [1.0, 1.0, 0.45, 1.0, 0.12]  # Active, Break, Depression, Orographic, WD

    rng = np.random.default_rng(seed)
    n_days = n_seasons * season_length
    predictors = np.zeros((n_days, N_PREDICTORS, grid, grid), dtype=np.float32)
    raw_nwp = np.zeros((n_days, grid, grid), dtype=np.float32)
    truth = np.zeros((n_days, grid, grid), dtype=np.float32)
    regime = np.zeros(n_days, dtype=np.int64)
    season_id = np.zeros(n_days, dtype=np.int64)
    subdaily_predictors = np.zeros(
        (n_days, n_subdaily_steps, N_PREDICTORS, grid, grid), dtype=np.float32)
    subdaily_regime = np.zeros((n_days, n_subdaily_steps), dtype=np.int64)

    idx = 0
    for s in range(n_seasons):
        seq = _markov_regime_sequence(season_length, N_REGIMES, rng, regime_prior=regime_prior)
        weights = smooth_regime_weights(seq, N_REGIMES, window=5)
        for d in range(season_length):
            w = weights[d]
            order = np.argsort(-w)
            dominant, secondary = order[0], order[1]
            r = int(dominant)  # discrete label kept only for reporting/validation

            t = true_rainfall_pattern(dominant, grid, rng)
            if w[secondary] > 0.15:  # near a transition -> blend rainfall pattern too
                t_sec = true_rainfall_pattern(secondary, grid, rng)
                blend = w[secondary] / (w[dominant] + w[secondary])
                t = (1 - blend) * t + blend * t_sec

            raw = raw_nwp_forecast(t, dominant, grid, rng)
            pred = predictor_fields_from_weights(w, grid, rng)

            predictors[idx] = pred
            raw_nwp[idx] = raw
            truth[idx] = t
            regime[idx] = r
            season_id[idx] = s

            # Genuine sub-daily trajectory for this day (clamped at season edges).
            w_prev = weights[d - 1] if d > 0 else w
            w_next = weights[d + 1] if d < season_length - 1 else w
            sub_traj, subdaily_hours = _subdaily_weight_trajectory(
                w_prev, w, w_next, n_subdaily_steps, rng, wobble_std=subdaily_wobble_std)
            for step in range(n_subdaily_steps):
                subdaily_predictors[idx, step] = predictor_fields_from_weights(
                    sub_traj[step], grid, rng)
                subdaily_regime[idx, step] = int(np.argmax(sub_traj[step]))

            idx += 1

    return {
        "predictors": predictors,
        "raw_nwp": raw_nwp,
        "truth": truth,
        "regime": regime,
        "season_id": season_id,
        "regime_names": REGIME_NAMES,
        "predictor_names": PREDICTOR_CHANNELS,
        "subdaily_predictors": subdaily_predictors,
        "subdaily_regime": subdaily_regime,
        "subdaily_hours": subdaily_hours,
    }


if __name__ == "__main__":
    import os
    ds = generate_dataset()
    os.makedirs("data/sample", exist_ok=True)
    np.savez_compressed(
        "data/sample/synthetic_dataset.npz",
        predictors=ds["predictors"], raw_nwp=ds["raw_nwp"], truth=ds["truth"],
        regime=ds["regime"], season_id=ds["season_id"],
        subdaily_predictors=ds["subdaily_predictors"], subdaily_regime=ds["subdaily_regime"],
        subdaily_hours=ds["subdaily_hours"],
    )
    print("Saved data/sample/synthetic_dataset.npz with", ds["predictors"].shape[0], "days",
          "and", ds["subdaily_predictors"].shape[1], "sub-daily steps/day")
