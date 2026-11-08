"""Sampling helpers."""
from __future__ import annotations

import numpy as np


def random_prefix(image_ids, done: set, seed: int) -> list:
    """Longest prefix of a seeded random order over `image_ids` that is fully in `done`.

    Used when a long GPU run is stopped early: images processed out of that order
    (e.g. an earlier id-sorted pass) count only if they fall inside the prefix, so the
    returned set is a simple random subsample of `image_ids`."""
    ids = sorted(image_ids)
    order = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    k = 0
    while k < len(order) and order[k] in done:
        k += 1
    return order[:k]
