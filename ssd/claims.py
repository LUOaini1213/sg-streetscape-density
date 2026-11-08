"""Every number quoted in README.md / paper/paper.md, recomputed from committed outputs.

Documents tag numbers as  <!--n:key-->VALUE<!--/n-->  ; check_numbers() recomputes
each key from outputs/*.json and fails on any mismatch or unknown key.
Untagged numbers are not checked, so every result number in the docs is tagged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TAG = re.compile(r"<!--n:([a-z0-9_]+)-->(.*?)<!--/n-->")


def _j(root: Path, name: str) -> dict:
    return json.loads((root / "outputs" / name).read_text())


def registry(root: Path) -> dict[str, str]:
    ag = _j(root, "agreement.json")
    pts = _j(root, "points_summary.json")
    win = _j(root, "window_models.json")
    isl = _j(root, "island_models.json")
    pw = _j(root, "power.json")
    kv = _j(root, "kv_islandwide_summary.json")
    f3, f2, pct = "{:.3f}".format, "{:.2f}".format, "{:.0%}".format
    r: dict[str, str] = {
        "gs_sg_images": f"{pts['n_points_on_land']:,}",
        "gs_kv_images": f"{pts['n_kartaview']:,}",
        "gs_mly_images": f"{pts['n_mapillary']:,}",
        "win_images_used": f"{win['n_images_used']:,}",
        "win_cells": str(win["n_cells"]),
        "win_blocks": str(win["n_blocks"]),
        "agree_n": str(ag["n_images"]),
        "isl_api_photos": f"{kv['api_photos_unique']:,}",
        "isl_images": f"{isl['n_images']:,}",
        "isl_cells": f"{isl['n_cells']:,}",
        "isl_pas": str(isl["n_planning_areas"]),
        "isl_blocks": str(isl["n_blocks"]),
        "n_stimuli": str(pw["n_stimuli"]),
        "win_corr_park": f2(win["corr_density_park"]),
        "isl_corr_park": f2(isl["corr_density_park"]),
        "win_soft_n": str(win["softened"]["n_candidates"]),
        "win_soft_hi": str(win["softened"]["n_high_density"]),
        "isl_soft_n": str(isl["softened"]["n_candidates"]),
        "isl_soft_hi": str(isl["softened"]["n_high_density"]),
    }
    for k in ("gvi", "svi", "building_share"):
        for pair in ("gs_vs_m2f", "gs_vs_segf", "m2f_vs_segf"):
            r[f"agree_{k}_{pair}_ccc"] = f3(ag[k][pair]["ccc"])
            r[f"agree_{k}_{pair}_mae"] = f3(ag[k][pair]["mae"])
            r[f"agree_{k}_{pair}_bias"] = f"{ag[k][pair]['bias']:+.3f}"
    wmap = {"gvi": "green_view_index", "svi": "sky_view_index", "building_share": "building_share"}
    for k, col in wmap.items():
        for spec in ("density_only", "adjusted", "adjusted_nolanded", "adjusted_ols_clustered"):
            m = win["mixed"][col][spec]
            r[f"win_{k}_{spec}_b"] = f"{m['params']['log_gfa']:+.3f}"
            r[f"win_{k}_{spec}_ci"] = f"[{m['ci_low']['log_gfa']:+.3f}, {m['ci_high']['log_gfa']:+.3f}]"
        for fs, v in win["cv"][col].items():
            r[f"win_{k}_cv_{fs.replace('+', '_')}"] = f2(v["cv_r2_mean"])
        r[f"win_{k}_moran"] = f2(win["cell_ols"][col]["resid_moran_i"])
        r[f"win_{k}_moran_p"] = f3(win["cell_ols"][col]["resid_moran_p"])
        for spec in ("density_only", "adjusted"):
            m = isl["image_level"][k][spec]
            r[f"isl_{k}_{spec}_b"] = f"{m['params']['log_gfa']:+.3f}"
            r[f"isl_{k}_{spec}_ci"] = f"[{m['ci_low']['log_gfa']:+.3f}, {m['ci_high']['log_gfa']:+.3f}]"
        for fs, v in isl["cv"][k].items():
            r[f"isl_{k}_cv_{fs.replace('+', '_')}"] = f2(v["cv_r2_mean"])
        r[f"isl_{k}_moran"] = f2(isl["cell_ols"][k]["resid_moran_i"])
    g = pd.DataFrame(pw["grid"])
    for row in g.itertuples():
        r[f"power_e{int(round(row.effect * 100)):02d}_n{row.raters}_s{int(round(row.item_sd * 10)):02d}"] =             pct(row.power)
    return r


def check_numbers(root: Path, docs=("README.md", "paper/paper.md", "docs/protocol.md")) -> list[str]:
    reg = registry(root)
    errors = []
    for doc in docs:
        p = root / doc
        if not p.exists():
            errors.append(f"{doc}: missing")
            continue
        text = p.read_text(encoding="utf-8")
        tags = TAG.findall(text)
        if not tags:
            errors.append(f"{doc}: no tagged numbers")
        for key, val in tags:
            if key not in reg:
                errors.append(f"{doc}: unknown key {key}")
            elif val != reg[key]:
                errors.append(f"{doc}: {key} says {val!r}, outputs give {reg[key]!r}")
    return errors
