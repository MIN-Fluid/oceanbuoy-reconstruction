from __future__ import annotations

import argparse
import gzip
import io
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{stn}h{year}.txt.gz"
USER_AGENT = "oceanbuoy-reconstruction"
REQUEST_SLEEP_SEC = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 3.0
YEARS = list(range(2016, 2026))
REGIONS = {
    "socal": ["46011", "46054", "46053", "46069", "46025", "46028", "46218", "46219", "46251", "46259"],
    "gom": ["42001", "42002", "42019", "42020", "42035", "42040"],
    "pnw": ["46050", "46029", "46041", "46089"],
    "socal_added": ["46239", "46047"],
}


def download_one(stn: str, year: int, dest: Path) -> tuple[bool, str]:
    req = urllib.request.Request(BASE_URL.format(stn=stn, year=year), headers={"User-Agent": USER_AGENT})
    err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if not data:
                err = "empty response body"
            else:
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                        gz.read(1024)
                except OSError as e:
                    err = f"corrupt gzip: {e}"
                else:
                    dest.write_bytes(data)
                    return True, ""
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code == 404:
                break
        except urllib.error.URLError as e:
            err = f"URLError {e.reason}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SEC * attempt)
    return False, err


def download_all(stations: list[str], years: list[int]) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = {"generated": datetime.now(timezone.utc).isoformat(), "failures": [], "skipped_existing": [], "downloaded": []}
    total, done = len(stations) * len(years), 0
    for stn in stations:
        (RAW_DIR / stn).mkdir(parents=True, exist_ok=True)
        for year in years:
            done += 1
            dest = RAW_DIR / stn / f"{stn}h{year}.txt.gz"
            tag = f"[{done}/{total}] {stn} {year}"
            if dest.exists() and dest.stat().st_size > 0:
                print(f"{tag}: present, skipped", flush=True)
                log["skipped_existing"].append(f"{stn}-{year}")
                continue
            ok, err = download_one(stn, year, dest)
            if ok:
                print(f"{tag}: OK ({dest.stat().st_size / 1024:.0f} KB)", flush=True)
                log["downloaded"].append(f"{stn}-{year}")
            else:
                print(f"{tag}: FAILED ({err})", flush=True)
                log["failures"].append({"station": stn, "year": year, "error": err})
            time.sleep(REQUEST_SLEEP_SEC)
    (RAW_DIR / "download_log.json").write_text(json.dumps(log, indent=2))
    print(f"downloaded={len(log['downloaded'])} skipped={len(log['skipped_existing'])} failed={len(log['failures'])}")
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", choices=list(REGIONS), default=list(REGIONS))
    ap.add_argument("--stations", nargs="+", default=None)
    ap.add_argument("--years", nargs="+", type=int, default=YEARS)
    a = ap.parse_args()
    stations = a.stations or [s for r in a.regions for s in REGIONS[r]]
    download_all(stations, a.years)


if __name__ == "__main__":
    main()
