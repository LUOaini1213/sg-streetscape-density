"""Agreement statistics between two measurements of the same quantity."""
from __future__ import annotations

import numpy as np
from scipy import stats


def lins_ccc(a, b) -> float:
    """Lin's concordance correlation coefficient (penalises bias and scale shifts)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float(2 * cov / (va + vb + (ma - mb) ** 2))


def bland_altman(a, b) -> dict:
    """Mean difference (a - b) and 95% limits of agreement."""
    d = np.asarray(a, float) - np.asarray(b, float)
    bias, sd = d.mean(), d.std(ddof=1)
    return {"bias": float(bias), "loa_low": float(bias - 1.96 * sd), "loa_high": float(bias + 1.96 * sd)}


def agreement(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        raise ValueError("need at least 3 paired values")
    return {"n": int(len(a)),
            "pearson_r": float(stats.pearsonr(a, b)[0]),
            "spearman_rho": float(stats.spearmanr(a, b)[0]),
            "ccc": lins_ccc(a, b),
            "mae": float(np.abs(a - b).mean()),
            **bland_altman(a, b)}
