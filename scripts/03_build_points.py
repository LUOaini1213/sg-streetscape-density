"""Clean the Singapore Global Streetscapes subset into an analysis point table.

- keep images inside the URA MP2019 subzone polygons (drops sea / Johor / Batam);
- attach SVY21 coordinates, 250 m cell id, 2 km CV block, subzone & planning area;
- write a slim derived table (CC BY-SA 4.0, derived from Global Streetscapes)
  to data/derived/points.parquet, and a stratified KartaView sample list for
  our own segmentation run.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ssd.grid import block_id, cell_id

KEEP = ["uuid", "source", "orig_id", "lat", "lon", "year", "month", "hour", "width", "height",
        "heading", "platform", "quality", "view_direction", "lighting_condition", "weather",
        "glare", "reflection", "sequence_id", "highway", "type_highway", "lanes",
        "Building", "Wall", "Vegetation", "Terrain", "Sky", "Road", "Sidewalk", "Car", "Person",
        "Total", "green_view_index", "sky_view_index",
        "Safe", "Lively", "Beautiful", "Wealthy", "Boring", "Depressing"]
SAMPLE_N = 600
PER_CELL = 8
SEED = 20261106


def main() -> None:
    raw = pd.read_parquet("data/raw/gs_singapore.parquet")
    kv = pd.read_parquet("data/raw/gs_singapore_kv.parquet")[["uuid", "kv_fileurlLTh", "kv_user", "kv_shotDate"]]
    pts = gpd.GeoDataFrame(raw[KEEP], geometry=gpd.points_from_xy(raw.lon, raw.lat), crs=4326).to_crs(3414)
    sz = gpd.read_file("data/raw/mp19_subzone.geojson").to_crs(3414)[["SUBZONE_N", "PLN_AREA_N", "REGION_N", "geometry"]]
    pts = gpd.sjoin(pts, sz, how="inner", predicate="within").drop(columns="index_right")
    pts["x"] = pts.geometry.x.round(1)
    pts["y"] = pts.geometry.y.round(1)
    pts["cell"] = cell_id(pts.x.values, pts.y.values)
    pts["block"] = block_id(pts.x.values, pts.y.values)
    # pixel shares (Global Streetscapes stores pixel counts; Total = labelled pixels)
    pts["building_share"] = (pts["Building"] + pts["Wall"]) / pts["Total"]
    pts["road_share"] = pts["Road"] / pts["Total"]
    df = pd.DataFrame(pts.drop(columns="geometry"))
    df = df.drop(columns=["Building", "Wall", "Vegetation", "Terrain", "Sky", "Road", "Sidewalk", "Car", "Person"])
    out = Path("data/derived")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "points.parquet", index=False)

    # Stratified KartaView sample for our own segmentation: daytime, good quality,
    # at most PER_CELL images per 250 m cell, at most 2 per sequence per cell.
    cand = df[(df.source == "KartaView") & (df.lighting_condition == "day") & (df.quality == "good")]
    cand = cand.merge(kv, on="uuid", how="inner")
    rng = np.random.default_rng(SEED)
    cand = cand.iloc[rng.permutation(len(cand))]
    cand = cand.groupby(["cell", "sequence_id"]).head(2).groupby("cell").head(PER_CELL)
    sample = cand.iloc[rng.permutation(len(cand))[:SAMPLE_N]]
    sample[["uuid", "orig_id", "kv_fileurlLTh", "kv_user", "kv_shotDate", "lat", "lon", "cell",
            "green_view_index", "sky_view_index", "building_share"]].to_csv(out / "kv_sample.csv", index=False)
    summary = {"n_raw_bbox": int(len(raw)), "n_points_on_land": int(len(df)),
               "n_kartaview": int((df.source == "KartaView").sum()),
               "n_mapillary": int((df.source == "Mapillary").sum()),
               "n_cells_with_images": int(df.cell.nunique()),
               "n_kv_sample_candidates": int(len(cand)), "n_kv_sample": int(len(sample))}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/points_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
