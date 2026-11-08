"""Shared helpers for the analysis scripts."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ssd.density import cells_frame, coverage, landuse_metrics
from ssd.grid import CELL_M, cell_bounds


def read_cached(name: str) -> gpd.GeoDataFrame:
    """Read a raw GeoJSON once and cache it as GeoParquet in SVY21."""
    pq = Path("data/derived") / (Path(name).stem + ".parquet")
    if pq.exists():
        return gpd.read_parquet(pq)
    g = gpd.read_file(Path("data/raw") / name).to_crs(3414)
    g.to_parquet(pq)
    return g


def density_table(cell_ids, size=CELL_M) -> pd.DataFrame:
    cells = cells_frame(cell_ids, size)
    lu = read_cached("mp19_landuse.geojson")
    lu = lu[lu.intersects(cells.union_all())]
    hdb = read_cached("hdb_existing_building.geojson")
    hdb = hdb[hdb.intersects(cells.union_all())]
    out = landuse_metrics(cells, lu).join(coverage(cells, hdb, "hdb_coverage"))
    out["gfa_intensity_nolanded"] = landuse_metrics(cells, lu, landed=None)["gfa_intensity"]
    b = np.array([cell_bounds(c, size) for c in out.index])
    out["cx"], out["cy"] = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    return out
