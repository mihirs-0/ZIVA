"""Pilot-informed power analysis.

Estimates the number of paired scenarios needed in a confirmatory run to
detect a chosen paired valence effect, by simulation from the empirical pilot
distribution: pilot paired differences are recentered to zero (empirical
null), a hypothesized effect is added, samples of size n are drawn with
replacement, and the paired sign-flip test is applied. Approximate by design.
"""

from __future__ import annotations

import numpy as np


def simulate_power(
    pilot_diffs: np.ndarray,
    effect_points: float,
    n_pairs: int,
    alpha: float = 0.05,
    n_sim: int = 500,
    n_perm: int = 1000,
    seed: int = 0,
) -> float:
    diffs = np.asarray(pilot_diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 3:
        raise ValueError("need at least 3 pilot paired differences")
    centered = diffs - diffs.mean()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        sample = rng.choice(centered, size=n_pairs, replace=True) + effect_points
        mean = sample.mean()
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n_pairs))
        perm_means = (signs * sample).mean(axis=1)
        p = (np.sum(np.abs(perm_means) >= abs(mean)) + 1) / (n_perm + 1)
        if p < alpha:
            hits += 1
    return hits / n_sim


def required_pairs(
    pilot_diffs: np.ndarray,
    effect_points: float = 5.0,
    target_power: float = 0.8,
    alpha: float = 0.05,
    max_pairs: int = 2000,
    seed: int = 0,
) -> dict:
    """Smallest n (searched over a doubling grid then refined) reaching target power."""
    grid = [10, 20, 40, 80, 160, 320, 640, 1280, 2000]
    result_curve = []
    found = None
    for n in grid:
        if n > max_pairs:
            break
        p = simulate_power(pilot_diffs, effect_points, n, alpha=alpha, seed=seed)
        result_curve.append({"n_pairs": n, "power": round(p, 3)})
        if p >= target_power and found is None:
            found = n
            break
    # refine between previous grid point and found
    if found is not None and found > grid[0]:
        lo = grid[grid.index(found) - 1]
        hi = found
        while hi - lo > max(2, lo // 8):
            mid = (lo + hi) // 2
            p = simulate_power(pilot_diffs, effect_points, mid, alpha=alpha, seed=seed)
            result_curve.append({"n_pairs": mid, "power": round(p, 3)})
            if p >= target_power:
                hi = mid
            else:
                lo = mid
        found = hi
    return {
        "effect_points": effect_points,
        "target_power": target_power,
        "alpha": alpha,
        "estimated_required_pairs": found,
        "note": "approximate paired sign-flip simulation from the empirical pilot distribution",
        "curve": sorted(result_curve, key=lambda r: r["n_pairs"]),
    }
