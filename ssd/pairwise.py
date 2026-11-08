"""Pairwise-comparison survey: simulation, Bradley-Terry and TrueSkill scoring, power.

A comparison is a row (i, j, y) meaning item i was shown against item j and
y = 1 if i was chosen ("feels less crowded"), else 0. No real responses exist
in this repository; everything here runs on simulated data.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- simulation
def simulate(latent: np.ndarray, n_raters: int, pairs_per_rater: int, rng: np.random.Generator,
             rater_sd: float = 0.0) -> np.ndarray:
    """Simulate choices under a Bradley-Terry-logit model.

    latent: true item utilities (logit scale). Each rater sees random distinct pairs;
    rater_sd scales a per-rater multiplicative 'discrimination' (1 + N(0, rater_sd)).
    Returns int array of shape (n, 4): rater, i, j, y.
    """
    k = len(latent)
    latent = np.asarray(latent, float)
    n = n_raters * pairs_per_rater
    rater = np.repeat(np.arange(n_raters), pairs_per_rater)
    i = rng.integers(0, k, n)
    j = (i + rng.integers(1, k, n)) % k          # j != i, uniform over the other items
    disc = np.ones(n_raters)
    if rater_sd > 0:
        disc = np.maximum(0.05, 1.0 + rng.normal(0, rater_sd, n_raters))
    p = 1.0 / (1.0 + np.exp(-disc[rater] * (latent[i] - latent[j])))
    y = (rng.random(n) < p).astype(np.int64)
    return np.column_stack([rater, i, j, y]).astype(np.int64)


# ---------------------------------------------------------------- Bradley-Terry (MM)
def bradley_terry(comp: np.ndarray, n_items: int, prior: float = 0.5, iters: int = 500,
                  tol: float = 1e-9) -> np.ndarray:
    """Bradley-Terry log-strengths by Hunter's (2004) MM algorithm.

    `prior` adds that many virtual wins and losses of every item against a
    reference item of strength 1, which keeps estimates finite for items that
    won or lost all their comparisons. Returns log-strengths centred at 0.
    """
    comp = np.asarray(comp)
    i, j, y = comp[:, -3], comp[:, -2], comp[:, -1]
    winner = np.where(y == 1, i, j)
    wins = np.bincount(winner, minlength=n_items).astype(float) + prior
    p = np.ones(n_items)
    for _ in range(iters):
        denom = np.zeros(n_items)
        s = 1.0 / (p[i] + p[j])
        np.add.at(denom, i, s)
        np.add.at(denom, j, s)
        denom += 2 * prior / (p + 1.0)
        new = wins / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(p))) < tol:
            p = new
            break
        p = new
    lp = np.log(p)
    return lp - lp.mean()


# ---------------------------------------------------------------- TrueSkill (1 vs 1, no draws)
def _v(t: float) -> float:
    return stats.norm.pdf(t) / max(stats.norm.cdf(t), 1e-12)


def trueskill(comp: np.ndarray, n_items: int, mu0: float = 25.0, sigma0: float = 25.0 / 3,
              beta: float = 25.0 / 6, tau: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Sequential TrueSkill updates (Herbrich et al. 2007) for two-player, no-draw games."""
    mu = np.full(n_items, mu0, float)
    var = np.full(n_items, sigma0 ** 2, float)
    for row in np.asarray(comp):
        a, b, y = row[-3], row[-2], row[-1]
        w, l = (a, b) if y == 1 else (b, a)
        var[w] += tau ** 2
        var[l] += tau ** 2
        c = math.sqrt(2 * beta ** 2 + var[w] + var[l])
        t = (mu[w] - mu[l]) / c
        v = _v(t)
        wv = v * (v + t)
        mu[w] += var[w] / c * v
        mu[l] -= var[l] / c * v
        var[w] *= max(1 - var[w] / c ** 2 * wv, 1e-6)
        var[l] *= max(1 - var[l] / c ** 2 * wv, 1e-6)
    return mu, np.sqrt(var)


# ---------------------------------------------------------------- analysis & power
def slope_test(scores: np.ndarray, x: np.ndarray, covars: np.ndarray | None = None) -> tuple[float, float]:
    """OLS of item scores on standardised feature x (+ optional covariates).
    Returns (slope, two-sided p-value) for x. Item-level inference: n = number of items."""
    x = (x - x.mean()) / x.std()
    cols = [np.ones_like(x), x]
    if covars is not None:
        c = np.atleast_2d(covars.T).T
        cols += [c[:, k] for k in range(c.shape[1])]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, scores, rcond=None)
    resid = scores - X @ beta
    dof = len(scores) - X.shape[1]
    s2 = resid @ resid / dof
    se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    t = beta[1] / se
    return float(beta[1]), float(2 * stats.t.sf(abs(t), dof))


def power(feature: np.ndarray, effect: float, item_sd: float, n_raters: int, pairs_per_rater: int,
          n_sims: int = 200, rater_sd: float = 0.3, alpha: float = 0.05, seed: int = 0) -> float:
    """Simulated power of the two-stage test (BT scores -> OLS on feature).

    Latent utility = effect * z(feature) + N(0, item_sd): `item_sd` is the scene-level
    appeal not explained by the feature, which bounds power however many raters respond.
    """
    rng = np.random.default_rng(seed)
    z = (feature - feature.mean()) / feature.std()
    hits = 0
    for _ in range(n_sims):
        latent = effect * z + rng.normal(0, item_sd, len(z))
        comp = simulate(latent, n_raters, pairs_per_rater, rng, rater_sd)
        s = bradley_terry(comp, len(z))
        b, p = slope_test(s, feature)
        hits += (p < alpha) and (b > 0)
    return hits / n_sims
