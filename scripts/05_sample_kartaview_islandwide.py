"""Island-wide KartaView sample via the public KartaView API (no account/token).

Queries a hexagonal lattice of points (spacing 1 km, search radius 500 m)
inside the URA MP2019 subzones, keeps 08:00-17:59 local-time, post-2015, non-panoramic photos, and
stratifies to at most PER_CELL photos per 250 m cell from distinct sequences.
Output: data/derived/kv_islandwide_sample.csv (metadata only; CC BY-SA 4.0
attribution = KartaView contributors, sequence id recorded per photo).
"""
import concurrent.futures as cf
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from ssd.grid import block_id, cell_id

API = "https://api.kartaview.org/2.0/photo/"
SPACING = 1000.0
RADIUS = 500
PER_CELL = 2
MAX_N = 2600
SEED = 7
RAW = Path("data/raw/kv_api_cache.jsonl")


def lattice(poly_union, spacing: float) -> np.ndarray:
    minx, miny, maxx, maxy = poly_union.bounds
    pts = []
    for r, y in enumerate(np.arange(miny, maxy, spacing * np.sqrt(3) / 2)):
        off = (spacing / 2) * (r % 2)
        for x in np.arange(minx + off, maxx, spacing):
            pts.append((x, y))
    g = gpd.GeoSeries(gpd.points_from_xy(*np.array(pts).T), crs=3414)
    return g[g.within(poly_union)]


def query(lat: float, lng: float) -> list:
    for attempt in range(5):
        try:
            r = requests.get(API, params={"lat": lat, "lng": lng, "radius": RADIUS, "itemsPerPage": 100},
                             timeout=60)
            if r.status_code == 200:
                return r.json()["result"]["data"]
        except (requests.RequestException, ValueError, KeyError):
            pass
        time.sleep(3 * (attempt + 1))
    return []


def main() -> None:
    sz = gpd.read_file("data/raw/mp19_subzone.geojson").to_crs(3414)
    land = sz.union_all() if hasattr(sz, "union_all") else sz.unary_union
    pts = lattice(land, SPACING).to_crs(4326)
    done = set()
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            done.add(tuple(json.loads(line)["q"]))
    todo = [(round(p.y, 6), round(p.x, 6)) for p in pts]
    todo = [q for q in todo if q not in done]
    with RAW.open("a") as fh, cf.ThreadPoolExecutor(4) as ex:
        for i, (q, data) in enumerate(zip(todo, ex.map(lambda q: query(*q), todo))):
            fh.write(json.dumps({"q": q, "data": data}) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(i, len(todo), flush=True)
    qset = {(round(p.y, 6), round(p.x, 6)) for p in pts}
    rows = {}
    for line in RAW.read_text().splitlines():
        rec = json.loads(line)
        if tuple(rec["q"]) not in qset:
            continue
        for d in rec["data"]:
            rows[d["id"]] = d
    df = pd.DataFrame(rows.values())
    df["lat"] = df.lat.astype(float)
    df["lon"] = df.lng.astype(float)
    df["shot"] = pd.to_datetime(df.shotDate, errors="coerce")
    # KartaView shotDate is UTC; Singapore local time = UTC+8
    df["hour"] = (df.shot.dt.hour + 8) % 24
    n_api = len(df)
    df = df[(df.visibility == "public") & (df.status == "active") & (df.shot.dt.year >= 2015)
            & df.hour.between(8, 17) & (df.projection == "PLANE")]
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(3414)
    g = gpd.sjoin(g, sz[["SUBZONE_N", "PLN_AREA_N", "geometry"]], predicate="within").drop(columns="index_right")
    g["x"], g["y"] = g.geometry.x.round(1), g.geometry.y.round(1)
    g["cell"] = cell_id(g.x.values, g.y.values)
    g["block"] = block_id(g.x.values, g.y.values)
    rng = np.random.default_rng(SEED)
    g = g.iloc[rng.permutation(len(g))]
    g = g.drop_duplicates(["cell", "sequenceId"]).groupby("cell").head(PER_CELL)
    g = g.iloc[rng.permutation(len(g))[:MAX_N]]
    out = g[["id", "sequenceId", "lat", "lon", "x", "y", "cell", "block", "SUBZONE_N", "PLN_AREA_N",
             "shotDate", "heading", "autoImgProcessingResult", "fileurlLTh"]].rename(columns={"id": "kv_id"})
    out.to_csv("data/derived/kv_islandwide_sample.csv", index=False)
    summary = {"lattice_points": int(len(pts)), "api_photos_unique": int(n_api),
               "after_filters": int(len(df)), "sample": int(len(out)), "cells": int(out.cell.nunique()),
               "planning_areas": int(out.PLN_AREA_N.nunique())}
    Path("outputs/kv_islandwide_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
