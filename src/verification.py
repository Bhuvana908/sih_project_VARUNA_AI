"""
Step 5 (part): Standard IMD/WMO-style forecast verification metrics.
RMSE, ETS (Equitable Threat Score), CSI (Critical Success Index),
POD (Probability of Detection), FAR (False Alarm Ratio),
FSS (Fractions Skill Score).
"""
import numpy as np


def rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def contingency_counts(pred, obs, threshold):
    p = pred >= threshold
    o = obs >= threshold
    hits = np.sum(p & o)
    misses = np.sum((~p) & o)
    false_alarms = np.sum(p & (~o))
    correct_neg = np.sum((~p) & (~o))
    return hits, misses, false_alarms, correct_neg


def pod(pred, obs, threshold):
    h, m, f, c = contingency_counts(pred, obs, threshold)
    return float(h / (h + m)) if (h + m) > 0 else np.nan


def far(pred, obs, threshold):
    h, m, f, c = contingency_counts(pred, obs, threshold)
    return float(f / (h + f)) if (h + f) > 0 else np.nan


def csi(pred, obs, threshold):
    h, m, f, c = contingency_counts(pred, obs, threshold)
    denom = h + m + f
    return float(h / denom) if denom > 0 else np.nan


def ets(pred, obs, threshold):
    h, m, f, c = contingency_counts(pred, obs, threshold)
    total = h + m + f + c
    if total == 0:
        return np.nan
    hits_random = (h + m) * (h + f) / total
    denom = (h + m + f - hits_random)
    return float((h - hits_random) / denom) if denom > 0 else np.nan


def fss(pred, obs, threshold, neighborhood=3):
    """Fractions Skill Score over a square neighborhood window."""
    p_bin = (pred >= threshold).astype(float)
    o_bin = (obs >= threshold).astype(float)

    def neighborhood_frac(binary_field, k):
        pad = k // 2
        padded = np.pad(binary_field, pad, mode="edge")
        h, w = binary_field.shape
        out = np.zeros_like(binary_field)
        cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant")
        for i in range(h):
            for j in range(w):
                y0, y1 = i, i + k
                x0, x1 = j, j + k
                total = (cumsum[y1, x1] - cumsum[y0, x1] - cumsum[y1, x0] + cumsum[y0, x0])
                out[i, j] = total / (k * k)
        return out

    frac_p = neighborhood_frac(p_bin, neighborhood)
    frac_o = neighborhood_frac(o_bin, neighborhood)

    num = np.mean((frac_p - frac_o) ** 2)
    denom = np.mean(frac_p ** 2) + np.mean(frac_o ** 2)
    if denom == 0:
        return 1.0  # both fields entirely below threshold everywhere -> perfect (trivial) agreement
    return float(1 - num / denom)


def full_report(pred_stack, obs_stack, thresholds, neighborhood=3):
    """pred_stack, obs_stack: (N, H, W) arrays across many days.
    Returns a dict of metrics, aggregated over all days for each threshold,
    plus overall RMSE."""
    report = {"RMSE": rmse(pred_stack, obs_stack)}
    for th in thresholds:
        pods, fars, csis, etss, fsss = [], [], [], [], []
        for i in range(pred_stack.shape[0]):
            pods.append(pod(pred_stack[i], obs_stack[i], th))
            fars.append(far(pred_stack[i], obs_stack[i], th))
            csis.append(csi(pred_stack[i], obs_stack[i], th))
            etss.append(ets(pred_stack[i], obs_stack[i], th))
            fsss.append(fss(pred_stack[i], obs_stack[i], th, neighborhood))
        def safe_nanmean(vals):
            vals = np.asarray(vals, dtype=float)
            if np.all(np.isnan(vals)):
                return float("nan")  # threshold never exceeded by forecast+obs on any day
            return float(np.nanmean(vals))

        report[f"POD_{th}mm"] = safe_nanmean(pods)
        report[f"FAR_{th}mm"] = safe_nanmean(fars)
        report[f"CSI_{th}mm"] = safe_nanmean(csis)
        report[f"ETS_{th}mm"] = safe_nanmean(etss)
        report[f"FSS_{th}mm"] = safe_nanmean(fsss)
    return report


def per_regime_report(pred_stack, obs_stack, dominant_regime, n_regimes, regime_names,
                       thresholds, neighborhood=3, min_days=3):
    """
    Groups test-set days by their dominant regime (an (N,) int array, e.g.
    regime_probs.argmax(axis=1)) and runs full_report separately within each
    regime's subset of days.

    Regimes with fewer than `min_days` test-set days are skipped rather than
    reported on a handful of samples that would look precise but aren't —
    rare regimes (Monsoon Depression, Western Disturbance) may simply not
    have enough held-out test days to break out reliably.
    """
    reports = {}
    for r in range(n_regimes):
        mask = dominant_regime == r
        n = int(mask.sum())
        if n < min_days:
            continue
        reports[regime_names[r]] = {
            "n_days": n,
            "report": full_report(pred_stack[mask], obs_stack[mask], thresholds, neighborhood),
        }
    return reports
