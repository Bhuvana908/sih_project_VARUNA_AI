"""
Impact Index — reframes the exceedance-probability map from src/uncertainty.py
as an impact-based product rather than a raw probability.

MoES/IMD have been moving national policy from "how much rain" forecasting
towards Impact Based Forecasting (IBF): the same rainfall probability means a
very different thing for a flood-prone, densely populated district than for
a sparsely populated one with no drainage-strain history. This module
combines P(rainfall >= threshold) with a district vulnerability/exposure
layer to produce a single Action Confidence Index (ACI) per grid cell, plus
a categorical alert level, instead of leaving that translation to whoever
reads the raw probability map.

This does not require new atmospheric data. It requires one static
vulnerability layer per district/grid-cell (flood history, population
density, drainage capacity, low-lying terrain — whatever composite index is
available). No such dataset is wired up here; see
`load_placeholder_vulnerability_map` for a clearly-labelled stand-in, and
swap it for a real layer (e.g. built from Census population density +
CWC flood-prone-area records) before presenting this as more than a proof of
concept.
"""
import numpy as np


ALERT_LEVELS = ["Watch", "Alert", "Warning", "Severe"]
ALERT_THRESHOLDS = [25.0, 50.0, 75.0]   # ACI score boundaries -> next alert level up


def load_placeholder_vulnerability_map(grid_size, seed=42):
    """
    Stand-in for a real district vulnerability/exposure layer. Returns a
    (grid_size, grid_size) array in [0, 1], smoothly varying so neighbouring
    cells have similar vulnerability (real exposure data — population
    density, flood-plain proximity — is spatially smooth too, not random
    noise). Replace this function with a loader for real district-level
    vulnerability data; keep the same (grid_size, grid_size), [0, 1] output.
    """
    rng = np.random.RandomState(seed)
    base = rng.rand(grid_size, grid_size)
    # cheap smoothing so vulnerability varies gradually across the grid,
    # rather than cell-to-cell noise
    for _ in range(3):
        base = (base + np.roll(base, 1, axis=0) + np.roll(base, -1, axis=0)
                + np.roll(base, 1, axis=1) + np.roll(base, -1, axis=1)) / 5.0
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    return base


def compute_action_confidence_index(exceedance_probability_map, vulnerability_map,
                                     hazard_weight=0.6, vulnerability_weight=0.4):
    """
    Combines the model's own P(rainfall >= threshold) map (from
    src/uncertainty.py's `exceedance_probability`) with a static
    vulnerability map into a single 0-100 Action Confidence Index per cell.

    hazard_weight / vulnerability_weight let a district weight "how likely"
    against "how bad if it happens" differently; both default to a fixed
    60/40 split and should be tuned with domain experts, not treated as a
    result in themselves.
    """
    assert exceedance_probability_map.shape == vulnerability_map.shape, \
        "exceedance_probability_map and vulnerability_map must be on the same grid"
    combined = (hazard_weight * exceedance_probability_map
                + vulnerability_weight * vulnerability_map)
    return np.clip(combined * 100.0, 0.0, 100.0)


def alert_level_map(action_confidence_index):
    """
    Maps the continuous ACI (0-100) to one of ALERT_LEVELS per cell, using
    ALERT_THRESHOLDS as the boundaries. Returns an integer array of the same
    shape, where the integer indexes into ALERT_LEVELS.
    """
    levels = np.zeros(action_confidence_index.shape, dtype=np.int64)
    for boundary in ALERT_THRESHOLDS:
        levels += (action_confidence_index >= boundary).astype(np.int64)
    return levels


def summarize_district_alert(action_confidence_index, vulnerability_map,
                              district_mask, district_name):
    """
    Reduces the grid-level ACI down to one number + one alert level for a
    named district, given a boolean mask of which grid cells belong to it.
    Uses the max ACI within the district (worst-case cell), since a single
    severely-affected pocket within a district is what triggers action, not
    the district-wide average.
    """
    if not district_mask.any():
        return None
    district_aci = float(action_confidence_index[district_mask].max())
    level_idx = int(alert_level_map(np.array([[district_aci]]))[0, 0])
    return {
        "district": district_name,
        "action_confidence_index": round(district_aci, 1),
        "alert_level": ALERT_LEVELS[level_idx],
        "mean_vulnerability": round(float(vulnerability_map[district_mask].mean()), 2),
    }
