"""Density models: image-level mixed models and cell-level blocked-CV comparisons."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ssd.spatial import blocked_folds, cv_r2, knn_weights, morans_i_perm


def ols_fit_predict(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ b


def mean_fit_predict(Xtr, ytr, Xte):
    return np.full(len(Xte), ytr.mean())


def cv_compare(cells: pd.DataFrame, y: str, feature_sets: dict[str, list[str]], block_col: str,
               n_folds: int = 5, seeds=range(10)) -> dict:
    """Spatially blocked CV R^2 for several OLS feature sets, averaged over fold seeds."""
    res = {}
    for name, cols in feature_sets.items():
        vals = []
        for s in seeds:
            folds = blocked_folds(cells[block_col].values, n_folds, seed=s)
            X = cells[cols].to_numpy(float) if cols else np.zeros((len(cells), 0))
            fp = ols_fit_predict if cols else mean_fit_predict
            vals.append(cv_r2(X, cells[y].to_numpy(float), folds, fp))
        res[name] = {"cv_r2_mean": float(np.mean(vals)), "cv_r2_sd": float(np.std(vals))}
    return res


def ols_with_moran(cells: pd.DataFrame, formula: str, xy_cols=("cx", "cy"), k: int = 6) -> dict:
    """OLS (HC3 SEs) + Moran's I of residuals on k-NN weights."""
    m = smf.ols(formula, data=cells).fit(cov_type="HC3")
    w = knn_weights(cells[list(xy_cols)].to_numpy(), k=min(k, len(cells) - 1))
    mi, p = morans_i_perm(m.resid.to_numpy(), w, n_perm=499)
    return {"params": m.params.to_dict(), "ci_low": m.conf_int()[0].to_dict(),
            "ci_high": m.conf_int()[1].to_dict(), "pvalues": m.pvalues.to_dict(),
            "r2": float(m.rsquared), "n": int(m.nobs), "resid_moran_i": mi, "resid_moran_p": p}


def mixed_model(df: pd.DataFrame, formula: str, group: str) -> dict:
    """Linear mixed model with a random intercept per group (e.g. 250 m cell)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # default optimiser chain (bfgs -> ...); 'lbfgs' alone failed to give a PD Hessian here
        m = smf.mixedlm(formula, df, groups=df[group]).fit(reml=True)
        if not m.converged:
            m = smf.mixedlm(formula, df, groups=df[group]).fit(reml=True, method="powell")
    ci = m.conf_int()
    fe = [k for k in m.fe_params.index]
    return {"params": {k: float(m.fe_params[k]) for k in fe},
            "ci_low": {k: float(ci.loc[k, 0]) for k in fe},
            "ci_high": {k: float(ci.loc[k, 1]) for k in fe},
            "pvalues": {k: float(m.pvalues[k]) for k in fe},
            "group_var": float(m.cov_re.iloc[0, 0]), "resid_var": float(m.scale),
            "converged": bool(m.converged), "n": int(m.nobs), "n_groups": int(df[group].nunique())}


def ols_clustered(df: pd.DataFrame, formula: str, group: str) -> dict:
    """Image-level OLS with cluster-robust (by cell) standard errors."""
    g = pd.factorize(df[group])[0]
    m = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": g})
    ci = m.conf_int()
    return {"params": m.params.to_dict(), "ci_low": ci[0].to_dict(), "ci_high": ci[1].to_dict(),
            "pvalues": m.pvalues.to_dict(), "n": int(m.nobs)}
