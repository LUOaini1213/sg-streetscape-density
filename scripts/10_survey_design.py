"""Survey stimulus selection and simulation-based power analysis.

Stimuli: 40 island-wide KartaView scenes (only photos KartaView reports as
face/plate-blurred), stratified by built-density quartile
(GFA intensity) x GVI tertile so that greenery and density are not collinear
in the stimulus set. Thumbnails (640 px) are written to docs/survey/img with
CC BY-SA 4.0 attribution to KartaView contributors.

Power: two-stage analysis (Bradley-Terry scores -> OLS on z(GVI) + z(density)),
simulated under the pre-registered model; NO real responses are used.
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from ssd.pairwise import bradley_terry, simulate, slope_test

N_STIMULI = 40
SEED = 11
EFFECTS = [0.0, 0.2, 0.35, 0.5]      # logit change in P(chosen as 'less crowded') per SD of GVI
RATERS = [25, 50, 100, 200]
PAIRS_PER_RATER = 20
ITEM_SD = 0.5                          # unexplained scene-level appeal (logit SD)
RATER_SD = 0.3
N_SIMS = 200


def select_stimuli(imgs: pd.DataFrame, cells: pd.DataFrame, rng) -> pd.DataFrame:
    d = imgs.merge(cells[["cell", "gfa_intensity", "park_share"]], on="cell")
    d = d[d.gfa_intensity > 0]
    d["dq"] = pd.qcut(d.gfa_intensity, 4, labels=False)
    d["gt"] = pd.qcut(d.gvi, 3, labels=False)
    per = N_STIMULI // 12 + 1
    pick = (d.iloc[rng.permutation(len(d))].drop_duplicates("cell")
            .groupby(["dq", "gt"]).head(per))
    return pick.iloc[rng.permutation(len(pick))[:N_STIMULI]].reset_index(drop=True)


def power_grid(gvi: np.ndarray, dens: np.ndarray, rng_seed: int) -> list[dict]:
    zg = (gvi - gvi.mean()) / gvi.std()
    zd = (dens - dens.mean()) / dens.std()
    out = []
    rng = np.random.default_rng(rng_seed)
    for eff in EFFECTS:
        for n in RATERS:
            hits = 0
            for _ in range(N_SIMS):
                # pre-registered generative model: density makes scenes feel more crowded,
                # greenery less; plus unexplained scene appeal and rater heterogeneity
                latent = eff * zg - 0.3 * zd + rng.normal(0, ITEM_SD, len(zg))
                comp = simulate(latent, n, PAIRS_PER_RATER, rng, RATER_SD)
                s = bradley_terry(comp, len(zg))
                b, p = slope_test(s, gvi, covars=zd[:, None])
                hits += (p < 0.05) and (b > 0)
            out.append({"effect": eff, "raters": n, "pairs_per_rater": PAIRS_PER_RATER,
                        "power": hits / N_SIMS})
            print(out[-1], flush=True)
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    imgs = pd.read_csv("outputs/island_images.csv", dtype={"kv_id": str})
    cells = pd.read_csv("outputs/island_cells.csv")
    meta = pd.read_csv("data/derived/kv_islandwide_sample.csv", dtype={"kv_id": str})[
        ["kv_id", "fileurlLTh", "autoImgProcessingResult"]]
    # only photos that KartaView's pipeline has already blurred (faces / licence plates)
    imgs = imgs.merge(meta, on="kv_id")
    imgs = imgs[imgs.autoImgProcessingResult == "BLURRED"]
    stim = select_stimuli(imgs, cells, rng)
    img_dir = Path("docs/survey/img")
    img_dir.mkdir(parents=True, exist_ok=True)
    src = Path("data/images/island")
    items = []
    for k, r in stim.iterrows():
        f = src / f"{r.kv_id}.jpg"
        im = Image.open(f) if f.exists() else Image.open(io.BytesIO(requests.get(r.fileurlLTh, timeout=60).content))
        im = im.convert("RGB")
        im.thumbnail((640, 360))
        name = f"s{k:02d}.jpg"
        im.save(img_dir / name, quality=78)
        items.append({"id": f"s{k:02d}", "file": f"img/{name}", "kartaview_photo_id": r.kv_id,
                      "kartaview_sequence": int(r.sequenceId), "planning_area": r.PLN_AREA_N.title(),
                      "shot_utc": r.shotDate, "gvi": round(float(r.gvi), 4), "svi": round(float(r.svi), 4),
                      "gfa_intensity": round(float(r.gfa_intensity), 3)})
    Path("docs/survey/stimuli.json").write_text(json.dumps(items, indent=1))
    gvi = np.array([i["gvi"] for i in items])
    dens = np.log1p(np.array([i["gfa_intensity"] for i in items]))
    grid = power_grid(gvi, dens, SEED)
    res = {"n_stimuli": len(items), "corr_gvi_logdensity_stimuli": float(np.corrcoef(gvi, dens)[0, 1]),
           "item_sd": ITEM_SD, "rater_sd": RATER_SD, "n_sims": N_SIMS, "grid": grid}
    Path("outputs/power.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
