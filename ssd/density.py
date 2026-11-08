"""Density proxies per analysis cell from URA Master Plan 2019 and HDB footprints."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from ssd.grid import CELL_M, cell_bounds


LANDED_GPR = 1.4  # assumed equivalent GPR for 'LND' (landed housing) parcels; see README


def parse_gpr(v, landed: float | None = None) -> float:
    """Numeric gross plot ratio. 'LND' (landed housing) maps to `landed` if given;
    'EVA' (evaluated case by case), 'SDP' and other codes map to NaN."""
    if landed is not None and str(v).strip().upper() == "LND":
        return float(landed)
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return float("nan")
    return f if 0 < f < 30 else float("nan")


def cells_frame(cell_ids, size: float = CELL_M, crs=3414) -> gpd.GeoDataFrame:
    ids = sorted(set(cell_ids))
    return gpd.GeoDataFrame({"cell": ids}, geometry=[box(*cell_bounds(c, size)) for c in ids], crs=crs)


def overlay_shares(cells: gpd.GeoDataFrame, polys: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Intersection pieces of polys with cells, with piece area (m^2)."""
    pieces = gpd.overlay(cells[["cell", "geometry"]], polys, how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.geometry.area
    return pieces


def landuse_metrics(cells: gpd.GeoDataFrame, lu: gpd.GeoDataFrame, landed: float | None = LANDED_GPR) -> pd.DataFrame:
    """Per cell: area-weighted mean GPR over parcels with a numeric GPR, coverage of
    those parcels, and land-use shares (residential, park/open space, road)."""
    lu = lu.copy()
    lu["gpr"] = lu["GPR"].map(lambda v: parse_gpr(v, landed))
    pieces = overlay_shares(cells, lu[["LU_DESC", "gpr", "geometry"]])
    area = cells.set_index("cell").geometry.area
    num = pieces.dropna(subset=["gpr"])
    g = num.assign(f=num.gpr * num.a).groupby("cell")
    out = pd.DataFrame({
        # area-weighted mean GPR over parcels that carry a numeric GPR
        "gpr_mean": g.f.sum() / g.a.sum(),
        "gpr_area_share": g.a.sum() / area,
        # gross floor-area intensity = sum(GPR * parcel area) / cell area
        "gfa_intensity": g.f.sum() / area,
    })
    desc = pieces.LU_DESC.fillna("").str.upper()
    for name, pat in {"res_share": "RESIDENTIAL", "park_share": "PARK|OPEN SPACE|RESERVE SITE|NATURE",
                      "road_share_lu": "^ROAD"}.items():
        out[name] = pieces[desc.str.contains(pat)].groupby("cell").a.sum() / area
    out = out.reindex(area.index)
    fill = ["gpr_area_share", "gfa_intensity", "res_share", "park_share", "road_share_lu"]
    out[fill] = out[fill].fillna(0.0)
    return out


def coverage(cells: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame, col: str) -> pd.Series:
    """Footprint area share of each cell (ground space index)."""
    pieces = overlay_shares(cells, footprints[["geometry"]])
    area = cells.set_index("cell").geometry.area
    return (pieces.groupby("cell").a.sum() / area).reindex(area.index).fillna(0.0).rename(col)
