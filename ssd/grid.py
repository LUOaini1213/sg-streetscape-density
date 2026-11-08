"""Square analysis grid in SVY21 (EPSG:3414) metres."""
from __future__ import annotations

import numpy as np

CELL_M = 250.0     # analysis cell size
BLOCK_M = 2000.0   # spatial CV block size


def cell_index(x: np.ndarray, y: np.ndarray, size: float = CELL_M) -> tuple[np.ndarray, np.ndarray]:
    """Integer column/row of the grid cell containing each point (floor division)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.floor(x / size).astype(np.int64), np.floor(y / size).astype(np.int64)


def cell_id(x, y, size: float = CELL_M) -> np.ndarray:
    """String id 'col_row' of the grid cell containing each point."""
    i, j = cell_index(x, y, size)
    return np.char.add(np.char.add(i.astype(str), "_"), j.astype(str))


def cell_bounds(cid: str, size: float = CELL_M) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of a cell id."""
    i, j = (int(v) for v in cid.split("_"))
    return i * size, j * size, (i + 1) * size, (j + 1) * size


def block_id(x, y, size: float = BLOCK_M) -> np.ndarray:
    """Coarser spatial block id used for blocked cross-validation."""
    return cell_id(x, y, size)
