from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as base

GRID = pd.date_range("2015-12-31 23:00", "2026-01-01 00:00", freq="1h", tz="UTC")
OUT = {r: (f"data/aligned/{r}.npz", f"data/aligned/{r}_stations.json") for r in ("socal", "gom", "pnw", "socal12")}


def build(region: str) -> None:
    meta = json.loads((ROOT / "configs" / "stations" / f"{region}.json").read_text())
    ids = list(meta)
    aligned = np.full((len(ids), len(GRID), len(base.VARIABLES)), np.nan, np.float32)
    for i, stn in enumerate(ids):
        aligned[i] = base.align_to_grid(base.load_station_raw(stn), GRID).values.astype(np.float32)
        print(f"  {region} {stn}: {int(np.isfinite(aligned[i]).sum())} / {aligned[i].size} non-NaN cells")
    npz, st = (ROOT / p for p in OUT[region])
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, stations=np.array(ids), timestamps=GRID.asi8 // 10 ** 9, variables=np.array(base.VARIABLES),
                        data=aligned, unknown_mask=np.isnan(aligned))
    st.write_text(json.dumps(meta, indent=1))
    print(f"{region}: {aligned.shape} -> {npz.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=list(OUT), choices=list(OUT))
    for region in ap.parse_args().regions:
        build(region)


if __name__ == "__main__":
    main()
