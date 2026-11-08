"""Streetscape indices from semantic label maps.

Definitions follow Global Streetscapes (Hou et al. 2024): green view index
(GVI) = vegetation pixels / labelled pixels; sky view index (SVI) = sky pixels /
labelled pixels. We add building_share = (building + wall) / labelled pixels.
Label names are normalised so that the same definitions work for Mapillary
Vistas ("Vegetation", "construction--structure--building" ...) and Cityscapes
("vegetation", "building", "wall") label sets.
"""
from __future__ import annotations

import numpy as np

CATEGORIES = {
    "vegetation": {"vegetation"},
    "sky": {"sky"},
    "building": {"building", "wall"},
}
# labels that are not scene content (camera mount / ego vehicle): excluded from
# the denominator, mirroring how Global Streetscapes' 'Total' excludes unlabelled pixels.
EXCLUDE = {"car-mount", "ego-vehicle", "unlabeled", "unlabelled"}


def norm(name: str) -> str:
    """Normalise a class label: last '--' component, lower case, spaces/underscores -> '-'."""
    return name.split("--")[-1].strip().lower().replace(" ", "-").replace("_", "-")


def category_ids(id2label: dict) -> dict[str, list[int]]:
    """Map each index category to the integer class ids that belong to it."""
    out = {k: [] for k in CATEGORIES}
    for i, name in id2label.items():
        n = norm(name)
        for cat, names in CATEGORIES.items():
            if n in names:
                out[cat].append(int(i))
    for cat, ids in out.items():
        if not ids:
            raise ValueError(f"no label found for category {cat!r}")
    return out


def excluded_ids(id2label: dict) -> list[int]:
    return [int(i) for i, name in id2label.items() if norm(name) in EXCLUDE]


def shares(label_map: np.ndarray, cat_ids: dict[str, list[int]], exclude: list[int] | None = None,
           n_classes: int | None = None) -> dict[str, float]:
    """Fraction of (non-excluded) pixels in each category, plus 'labelled' fraction."""
    lm = np.asarray(label_map).ravel()
    if lm.size == 0:
        raise ValueError("empty label map")
    n_classes = int(n_classes or (lm.max() + 1))
    counts = np.bincount(lm, minlength=n_classes).astype(float)
    excl = sum(counts[i] for i in (exclude or []) if i < len(counts))
    total = counts.sum() - excl
    if total <= 0:
        return {**{k: float("nan") for k in cat_ids}, "labelled": 0.0}
    res = {k: float(sum(counts[i] for i in ids if i < len(counts)) / total) for k, ids in cat_ids.items()}
    res["labelled"] = float(total / counts.sum())
    return res
