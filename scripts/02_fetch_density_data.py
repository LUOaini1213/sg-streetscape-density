"""Download open density proxies for Singapore from data.gov.sg.

All datasets are published under the Singapore Open Data Licence v1.0
(https://data.gov.sg/open-data-licence), which permits redistribution of
derived works with attribution. Raw files go to data/raw/ (git-ignored).
"""
import json
import time
from pathlib import Path

import requests

DATASETS = {
    "mp19_landuse.geojson": "d_90d86daa5bfaa371668b84fa5f01424f",      # URA Master Plan 2019 Land Use layer
    "mp19_subzone.geojson": "d_8594ae9ff96d0c708bc2af633048edfb",      # URA MP2019 Subzone Boundary (No Sea)
    "mp19_planning_area.geojson": "d_4765db0e87b9c86336792efe8a1f7a66",  # URA MP2019 Planning Area Boundary (No Sea)
    "hdb_existing_building.geojson": "d_16b157c52ed637edd6ba1232e026258d",  # HDB Existing Building footprints
    "hdb_property_info.csv": "d_17f5382f26140b1fdae0ba2ef6239d2f",     # HDB Property Information (max floor level)
}
POLL = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/poll-download"
META = "https://api-production.data.gov.sg/v2/public/api/datasets/{}/metadata"


def fetch(dataset_id: str, dest: Path) -> None:
    for attempt in range(10):
        r = requests.get(POLL.format(dataset_id), timeout=60)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        url = r.json()["data"]["url"]
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
        return
    raise RuntimeError(f"rate-limited too often: {dataset_id}")


def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, did in DATASETS.items():
        dest = out / name
        if not dest.exists():
            print("downloading", name)
            fetch(did, dest)
            time.sleep(3)
        meta = requests.get(META.format(did), timeout=60).json().get("data", {})
        manifest[name] = {"datasetId": did, "name": meta.get("name"),
                          "lastUpdatedAt": meta.get("lastUpdatedAt"),
                          "bytes": dest.stat().st_size}
    Path("data/raw/manifest_datagovsg.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
