"""Tier B: island-wide KartaView sample segmented by us (Mask2Former, Vistas).

At most 2 images per 250 m cell, so inference is at image level with
cluster-robust SEs (clusters = 2 km blocks, which also absorbs short-range
spatial dependence) plus cell-level OLS compared by 2 km blocked CV.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ssd.models import cv_compare, ols_clustered, ols_with_moran
from scripts.scripts_common import density_table

OUTCOMES = ["gvi", "svi", "building_share"]


def main() -> None:
    meta = pd.read_csv("data/derived/kv_islandwide_sample.csv", dtype={"kv_id": str})
    seg = pd.read_csv("data/seg/island_m2f.csv", dtype={"image_id": str}).drop_duplicates("image_id")
    df = meta.merge(seg, left_on="kv_id", right_on="image_id")
    df["year"] = pd.to_datetime(df.shotDate).dt.year
    cache = Path("data/derived/island_density.parquet")
    if cache.exists():
        dens = pd.read_parquet(cache)
    else:
        dens = density_table(df.cell.unique())
        dens.to_parquet(cache)
    df = df.join(dens, on="cell")
    df["log_gfa"] = np.log1p(df.gfa_intensity)
    df["yr"] = df.year - 2020
    ctrl = "park_share + res_share + road_share_lu + yr"
    res = {"n_images": int(len(df)), "n_cells": int(df.cell.nunique()), "n_blocks": int(df.block.nunique()),
           "n_planning_areas": int(df.PLN_AREA_N.nunique()), "image_level": {}, "cv": {}, "cell_ols": {}}
    for y in OUTCOMES:
        res["image_level"][y] = {
            "density_only": ols_clustered(df, f"{y} ~ log_gfa", "block"),
            "adjusted": ols_clustered(df, f"{y} ~ log_gfa + hdb_coverage + {ctrl}", "block"),
        }
    cells = df.groupby("cell").agg({**{y: "mean" for y in OUTCOMES}, "block": "first", "yr": "mean",
                                    "PLN_AREA_N": "first"}).join(dens)
    cells["log_gfa"] = np.log1p(cells.gfa_intensity)
    fsets = {"mean_only": [], "density": ["log_gfa", "hdb_coverage"],
             "controls": ["park_share", "res_share", "road_share_lu", "yr"],
             "density+controls": ["log_gfa", "hdb_coverage", "park_share", "res_share", "road_share_lu", "yr"]}
    for y in OUTCOMES:
        res["cv"][y] = cv_compare(cells, y, fsets, "block")
        res["cell_ols"][y] = ols_with_moran(cells, f"{y} ~ log_gfa + hdb_coverage + {ctrl}")
    res["corr_density_park"] = float(np.corrcoef(cells.log_gfa, cells.park_share)[0, 1])
    # softened density: top-quartile intensity cells whose GVI beats the density-only prediction
    m = smf.ols("gvi ~ log_gfa", data=cells).fit()
    cells["gvi_resid"] = m.resid
    q75 = cells.gfa_intensity.quantile(0.75)
    hi = cells[cells.gfa_intensity >= q75]
    soft = hi[(hi.gvi_resid > 0) & (hi.svi >= hi.svi.median())].sort_values("gvi_resid", ascending=False)
    res["softened"] = {"gfa_q75": float(q75), "n_high_density": int(len(hi)), "n_candidates": int(len(soft)),
                       "top_planning_areas": soft.head(10).PLN_AREA_N.value_counts().to_dict()}
    cells.reset_index().to_csv("outputs/island_cells.csv", index=False, float_format="%.5g")
    soft.reset_index().head(25)[["cell", "PLN_AREA_N", "gfa_intensity", "gvi", "svi", "building_share",
                                 "gvi_resid", "cx", "cy"]].to_csv("outputs/softened_candidates.csv",
                                                                   index=False, float_format="%.4g")
    df[["kv_id", "sequenceId", "cell", "PLN_AREA_N", "lat", "lon", "shotDate", "gvi", "svi",
        "building_share"]].to_csv("outputs/island_images.csv", index=False, float_format="%.5f")
    Path("outputs/island_models.json").write_text(json.dumps(res, indent=2, default=float))
    for y in OUTCOMES:
        a = res["image_level"][y]["adjusted"]
        print(y, round(a["params"]["log_gfa"], 4), (round(a["ci_low"]["log_gfa"], 4), round(a["ci_high"]["log_gfa"], 4)),
              {k: round(v["cv_r2_mean"], 3) for k, v in res["cv"][y].items()})
    print({k: v for k, v in res.items() if k not in ("image_level", "cell_ols", "cv")})


if __name__ == "__main__":
    main()
