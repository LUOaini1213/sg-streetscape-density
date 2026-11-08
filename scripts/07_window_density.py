"""Tier A: Global Streetscapes Singapore window (provided Mask2Former attributes).

Joins every image to its 250 m cell's density proxies (URA MP2019 GPR-derived
gross floor-area intensity, HDB footprint coverage, land-use shares) and fits
  (1) image-level linear mixed models with a random intercept per cell, and
  (2) cell-level OLS compared by spatially blocked CV (500 m blocks),
then lists 'softened density' candidate cells.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.scripts_common import density_table
from ssd.grid import cell_id
from ssd.models import cv_compare, mixed_model, ols_clustered, ols_with_moran

BLOCK_M = 500.0
MIN_IMAGES = 5
OUTCOMES = ["green_view_index", "sky_view_index", "building_share"]


def main() -> None:
    pts = pd.read_parquet("data/derived/points.parquet")
    n0 = len(pts)
    pts = pts[(pts.lighting_condition == "day") & (pts.quality != "very poor") & (pts.year >= 2015)].copy()
    counts = pts.groupby("cell").size()
    keep = counts[counts >= MIN_IMAGES].index
    pts = pts[pts.cell.isin(keep)].copy()
    cache = Path("data/derived/window_density.parquet")
    if cache.exists():
        dens = pd.read_parquet(cache)
    else:
        dens = density_table(pts.cell.unique())
        dens.to_parquet(cache)
    pts = pts.join(dens, on="cell")
    pts["log_gfa"] = np.log1p(pts.gfa_intensity)
    pts["walk"] = (pts.platform != "driving surface").astype(int)
    pts["kv"] = (pts.source == "KartaView").astype(int)
    pts["side"] = (pts.view_direction == "side").astype(int)
    pts["yr"] = pts.year - 2018
    pts["major_road"] = pts.highway.astype(str).str.contains("trunk|primary|motorway").astype(int)

    res = {"n_images_raw": int(n0), "n_images_used": int(len(pts)), "n_cells": int(pts.cell.nunique()),
           "min_images_per_cell": MIN_IMAGES, "mixed": {}, "cell_ols": {}, "cv": {}}
    ctrl = "park_share + res_share + walk + kv + side + major_road + yr"
    for y in OUTCOMES:
        res["mixed"][y] = {
            "density_only": mixed_model(pts, f"{y} ~ log_gfa", "cell"),
            "adjusted": mixed_model(pts, f"{y} ~ log_gfa + {ctrl}", "cell"),
            "adjusted_ols_clustered": ols_clustered(pts, f"{y} ~ log_gfa + {ctrl}", "cell"),
            "adjusted_nolanded": mixed_model(pts.assign(log_gfa=np.log1p(pts.gfa_intensity_nolanded)),
                                             f"{y} ~ log_gfa + {ctrl}", "cell"),
        }
    # cell level
    agg = {y: "mean" for y in OUTCOMES + ["Safe", "Lively", "Beautiful", "Depressing", "walk", "kv",
                                          "side", "major_road"]}
    cells = pts.groupby("cell").agg(agg).join(dens).join(pts.groupby("cell").size().rename("n_images"))
    cells["log_gfa"] = np.log1p(cells.gfa_intensity)
    cells["block"] = cell_id(cells.cx.values, cells.cy.values, BLOCK_M)
    fsets = {"mean_only": [], "density": ["log_gfa", "hdb_coverage"],
             "controls": ["park_share", "res_share", "walk", "kv", "side", "major_road"],
             "density+controls": ["log_gfa", "hdb_coverage", "park_share", "res_share", "walk", "kv",
                                  "side", "major_road"]}
    for y in OUTCOMES:
        res["cv"][y] = cv_compare(cells, y, fsets, "block")
        res["cell_ols"][y] = ols_with_moran(
            cells, f"{y} ~ log_gfa + hdb_coverage + park_share + res_share + walk + kv + side + major_road")
    for y in ["Safe", "Lively", "Beautiful", "Depressing"]:
        res["cell_ols"][f"perception_{y}"] = ols_with_moran(
            cells, f"{y} ~ log_gfa + green_view_index + sky_view_index + park_share + walk + kv")
    res["n_blocks"] = int(cells.block.nunique())
    res["corr_density_park"] = float(np.corrcoef(cells.log_gfa, cells.park_share)[0, 1])

    # 'softened density': top-quartile GFA intensity, and GVI above what density + view
    # controls predict (positive residual), ranked by residual.
    import statsmodels.formula.api as smf
    m = smf.ols("green_view_index ~ log_gfa + walk + kv + side + major_road", data=cells).fit()
    cells["gvi_resid"] = m.resid
    q75 = cells.gfa_intensity.quantile(0.75)
    soft = cells[(cells.gfa_intensity >= q75) & (cells.gvi_resid > 0)].sort_values("gvi_resid", ascending=False)
    res["softened"] = {"gfa_q75": float(q75), "n_high_density": int((cells.gfa_intensity >= q75).sum()),
                       "n_candidates": int(len(soft)), "top": soft.head(5).index.tolist()}
    Path("outputs").mkdir(exist_ok=True)
    cells.reset_index().to_csv("outputs/window_cells.csv", index=False, float_format="%.5g")
    Path("outputs/window_models.json").write_text(json.dumps(res, indent=2, default=float))
    for y in OUTCOMES:
        a = res["mixed"][y]["adjusted"]
        print(y, "adj log_gfa", round(a["params"]["log_gfa"], 4), (round(a["ci_low"]["log_gfa"], 4),
              round(a["ci_high"]["log_gfa"], 4)), {k: round(v["cv_r2_mean"], 3) for k, v in res["cv"][y].items()})
    print({k: v for k, v in res.items() if k not in ("mixed", "cell_ols", "cv")})


if __name__ == "__main__":
    main()
