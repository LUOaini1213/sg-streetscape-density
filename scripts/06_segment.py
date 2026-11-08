"""Run a pretrained semantic-segmentation model on downloaded street images.

Models (Hugging Face, used for inference only, weights not redistributed):
  m2f  facebook/mask2former-swin-large-mapillary-vistas-semantic (65 Vistas classes;
       same label set as Global Streetscapes' own segmentation)
  segf nvidia/segformer-b2-finetuned-cityscapes-1024-1024 (19 Cityscapes classes;
       an independent architecture + training set, for a model-sensitivity check)
GPU: fp16 on a 4 GB GTX 1650, batch size 1; images are the 1280x720 KartaView
large thumbnails. Writes one CSV row per image with GVI / SVI / building share.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (AutoImageProcessor, Mask2FormerForUniversalSegmentation,
                          SegformerForSemanticSegmentation)

from ssd.indices import category_ids, shares

MODELS = {
    "m2f": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    "segf": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
}


def load(key: str, device: str):
    name = MODELS[key]
    proc = AutoImageProcessor.from_pretrained(name)
    cls = Mask2FormerForUniversalSegmentation if key == "m2f" else SegformerForSemanticSegmentation
    model = cls.from_pretrained(name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    return proc, model.to(device).eval()


@torch.inference_mode()
def predict(key: str, proc, model, img: Image.Image, device: str) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    if key == "m2f":
        inputs = proc(images=img, return_tensors="pt")
        inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[img.size[::-1]])[0]
        return seg.cpu().numpy()
    inputs = proc(images=img, return_tensors="pt")
    logits = model(pixel_values=inputs["pixel_values"].to(device, dtype=dtype)).logits
    up = torch.nn.functional.interpolate(logits.float(), size=img.size[::-1], mode="bilinear", align_corners=False)
    return up.argmax(1)[0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("imgdir")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-masks", default="", help="optional dir for a few PNG label maps")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load(a.model, device)
    id2label = model.config.id2label
    cats = category_ids(id2label)
    files = sorted(Path(a.imgdir).glob("*.jpg"))
    if a.limit:
        files = files[: a.limit]
    done = set()
    out = Path(a.out)
    if out.exists():
        done = set(pd.read_csv(out)["image_id"].astype(str))
    rows, t0 = [], time.time()
    for k, f in enumerate(files):
        if f.stem in done:
            continue
        try:
            img = Image.open(f).convert("RGB")
        except OSError:
            continue
        seg = predict(a.model, proc, model, img, device)
        s = shares(seg, cats, n_classes=len(id2label))
        rows.append({"image_id": f.stem, "model": a.model, "w": img.size[0], "h": img.size[1],
                     "gvi": s["vegetation"], "svi": s["sky"], "building_share": s["building"]})
        if a.save_masks and k < 12:
            Path(a.save_masks).mkdir(parents=True, exist_ok=True)
            np.save(Path(a.save_masks) / f"{f.stem}_{a.model}.npy", seg.astype(np.uint8))
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            rows = []
            print(f"{k + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"done {len(files)} images in {time.time() - t0:.0f}s on {device}")


if __name__ == "__main__":
    main()
