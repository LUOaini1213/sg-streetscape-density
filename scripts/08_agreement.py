"""Agreement between Global Streetscapes' provided indices and our own segmentation.

Same 477 KartaView images (1280x720 thumbnails) run through
  m2f  : Mask2Former Swin-L, Mapillary Vistas (same label set as the dataset)
  segf : SegFormer-B2, Cityscapes (different architecture and training data)
Outputs outputs/agreement.json and outputs/agreement_images.csv (per-image
indices only - derived from CC BY-SA 4.0 data, released under CC BY-SA 4.0).
"""
import json
from pathlib import Path

import pandas as pd

from ssd.agreement import agreement

PAIRS = {"gvi": "gs_gvi", "svi": "gs_svi", "building_share": "gs_building_share"}


def main() -> None:
    gs = pd.read_csv("data/derived/kv_sample.csv").rename(columns={
        "uuid": "image_id", "green_view_index": "gs_gvi", "sky_view_index": "gs_svi",
        "building_share": "gs_building_share"})
    m2f = pd.read_csv("data/seg/gs_kv_m2f.csv").drop_duplicates("image_id")
    segf = pd.read_csv("data/seg/gs_kv_segf.csv").drop_duplicates("image_id")
    df = gs.merge(m2f, on="image_id").merge(segf, on="image_id", suffixes=("_m2f", "_segf"))
    res = {"n_images": int(len(df)), "n_cells": int(df.cell.nunique())}
    for ours, theirs in PAIRS.items():
        res[ours] = {
            "gs_vs_m2f": agreement(df[f"{ours}_m2f"], df[theirs]),
            "gs_vs_segf": agreement(df[f"{ours}_segf"], df[theirs]),
            "m2f_vs_segf": agreement(df[f"{ours}_segf"], df[f"{ours}_m2f"]),
        }
    Path("outputs/agreement.json").write_text(json.dumps(res, indent=2))
    cols = ["image_id", "orig_id", "cell"] + list(PAIRS.values()) + \
        [f"{k}_{m}" for k in PAIRS for m in ("m2f", "segf")]
    df[cols].to_csv("outputs/agreement_images.csv", index=False, float_format="%.5f")
    for k in PAIRS:
        print(k, {c: (round(v["pearson_r"], 3), round(v["ccc"], 3), round(v["bias"], 4), round(v["mae"], 4))
                  for c, v in res[k].items()})


if __name__ == "__main__":
    main()
