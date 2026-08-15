"""Paired statistical utilities: bootstrap CIs, sign-flip permutation tests,
standardized effect sizes. Deliberately simple and inspectable; no model
fitting is required for the primary result.
"""

from __future__ import annotations

import numpy as np


def paired_summary(diffs: np.ndarray, n_boot: int = 5000, n_perm: int = 5000,
                   seed: int = 0) -> dict:
    """Summary statistics for an array of paired differences.

    * mean with a scenario-level bootstrap percentile CI (resampling pairs);
    * two-sided sign-flip permutation p-value (exact null: no treatment effect
      within a pair implies the sign of each difference is exchangeable);
    * Cohen's d_z = mean(diff) / sd(diff).
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return {"n_pairs": 0, "mean": None, "ci95": [None, None], "p_perm": None, "cohens_dz": None}
    rng = np.random.default_rng(seed)
    mean = float(np.mean(diffs))

    if n == 1:
        return {"n_pairs": 1, "mean": round(mean, 4), "ci95": [round(mean, 4), round(mean, 4)],
                "p_perm": None, "cohens_dz": None}

    boots = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    ci = np.percentile(boots, [2.5, 97.5])

    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diffs).mean(axis=1)
    p = float((np.sum(np.abs(perm_means) >= abs(mean)) + 1) / (n_perm + 1))

    sd = float(np.std(diffs, ddof=1))
    dz = mean / sd if sd > 0 else None

    return {
        "n_pairs": n,
        "mean": round(mean, 4),
        "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "p_perm": round(p, 5),
        "cohens_dz": round(dz, 4) if dz is not None else None,
        "sd": round(sd, 4),
    }


def holm_bonferroni(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of secondary comparisons."""
    items = sorted(((k, v) for k, v in pvalues.items() if v is not None), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        adjusted[k] = round(running_max, 5)
    return adjusted


def pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)
