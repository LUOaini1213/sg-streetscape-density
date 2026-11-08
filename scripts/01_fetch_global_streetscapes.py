"""Fetch the Singapore subset of the Global Streetscapes tabular data.

Source: NUS-UAL/global-streetscapes on Hugging Face (CC BY-SA 4.0),
Hou et al. (2024) ISPRS J. Photogramm. Remote Sens. 215:216-238.

Uses DuckDB's HTTP range reads so that only the needed columns / row groups
are transferred. No account or token is required.
Output: data/raw/gs_singapore.parquet (git-ignored; re-create with this script).
"""
import argparse
from pathlib import Path

import duckdb

BASE = "https://huggingface.co/datasets/NUS-UAL/global-streetscapes/resolve/main/data/parquet/"
# Generous bounding box around Singapore; refined later with the city field.
BBOX = dict(lat_min=1.15, lat_max=1.48, lon_min=103.59, lon_max=104.10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql("SET http_timeout=600000; SET http_retries=10;")
    where = (f"lat BETWEEN {BBOX['lat_min']} AND {BBOX['lat_max']} "
             f"AND lon BETWEEN {BBOX['lon_min']} AND {BBOX['lon_max']}")
    main_pq = out / "gs_singapore.parquet"
    if not main_pq.exists():
        con.sql(f"COPY (SELECT * FROM '{BASE}streetscapes.parquet' WHERE {where}) "
                f"TO '{main_pq.as_posix()}' (FORMAT PARQUET)")
    n = con.sql(f"SELECT count(*) FROM '{main_pq.as_posix()}'").fetchone()[0]
    print("rows in bbox:", n)
    # KartaView metadata (image URLs) for the same uuids
    kv_pq = out / "gs_singapore_kv.parquet"
    if not kv_pq.exists():
        con.sql(f"COPY (SELECT k.* FROM '{BASE}metadata_kv.parquet' k "
                f"JOIN '{main_pq.as_posix()}' s USING (uuid)) TO '{kv_pq.as_posix()}' (FORMAT PARQUET)")
    print("kv rows:", con.sql(f"SELECT count(*) FROM '{kv_pq.as_posix()}'").fetchone()[0])


if __name__ == "__main__":
    main()
