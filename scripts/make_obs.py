from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, data as D, io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=C.ALL)
    region = ap.parse_args().region
    dcfg = C.load("data")
    ds = D.obs_from_aligned(region, dcfg)
    ds.attrs["meta"] = json.dumps(io.run_meta(dcfg, []))
    io.write_netcdf(ds, C.obs_unscreened(region))
    print(f"{region}: {ds.sizes['station']} stations x {ds.sizes['time']} hours -> {C.obs_unscreened(region)}")


if __name__ == "__main__":
    main()
