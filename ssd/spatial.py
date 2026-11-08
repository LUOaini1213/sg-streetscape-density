"""Spatial helpers: blocked cross-validation and Moran's I on k-NN weights."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def knn_weights(xy: np.ndarray, k: int = 6) -> np.ndarray:
    """Row-standardised k-nearest-neighbour weight matrix (dense; fine for n < 5000)."""
    xy = np.asarray(xy, float)
    n = len(xy)
    if n <= k:
        raise ValueError("need more points than k")
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    w = np.zeros((n, n))
    for i in range(n):
        nb = [j for j in idx[i] if j != i][:k]
        w[i, nb] = 1.0 / k
    return w


def morans_i(values: np.ndarray, w: np.ndarray) -> float:
    z = np.asarray(values, float) - np.mean(values)
    n, s0 = len(z), w.sum()
    return float((n / s0) * (z @ w @ z) / (z @ z))


def morans_i_perm(values, w, n_perm: int = 999, seed: int = 0) -> tuple[float, float]:
    """Moran's I with a one-sided (positive autocorrelation) permutation p-value."""
    rng = np.random.default_rng(seed)
    obs = morans_i(values, w)
    sims = np.array([morans_i(rng.permutation(values), w) for _ in range(n_perm)])
    p = (1 + np.sum(sims >= obs)) / (n_perm + 1)
    return obs, float(p)


def blocked_folds(blocks: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign whole spatial blocks to folds (no block is split across folds)."""
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {b: int(i % n_folds) for i, b in zip(range(len(uniq)), uniq[perm])}
    return np.array([fold_of[b] for b in blocks])


def cv_r2(X: np.ndarray, y: np.ndarray, folds: np.ndarray, fit_predict) -> float:
    """Out-of-fold R^2 = 1 - SSE/SST using pooled out-of-fold predictions."""
    y = np.asarray(y, float)
    pred = np.full_like(y, np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        pred[te] = fit_predict(X[tr], y[tr], X[te])
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
