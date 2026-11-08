"""Download KartaView 'large thumbnail' (1280x720) images listed in a CSV.

KartaView imagery is CC BY-SA 4.0; images are cached locally under
data/images/ (git-ignored) and are not redistributed in this repository,
except the small survey thumbnails under docs/survey/img (see their README).
"""
import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import pandas as pd
import requests


def get(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                dest.write_bytes(r.content)
                return True
            if r.status_code in (403, 404):
                return False
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("outdir")
    ap.add_argument("--id-col", default="uuid")
    ap.add_argument("--url-col", default="kv_fileurlLTh")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {row[a.id_col]: (row[a.url_col], out / f"{row[a.id_col]}.jpg") for _, row in df.iterrows()}
    ok = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(get, u, d): k for k, (u, d) in jobs.items()}
        for f in cf.as_completed(futs):
            ok += bool(f.result())
    print(f"downloaded {ok}/{len(jobs)} -> {out}")


if __name__ == "__main__":
    main()
