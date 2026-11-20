from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, data as D, io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=C.REGIONS)
    region = ap.parse_args().region
    dcfg, ecfg = C.load("data"), C.load("eval")
    obs, stations, _, _, _ = D.open_obs(C.obs_unscreened(region))
    w = np.isfinite(obs[:, :, dcfg["variables"].index("WVHT")].astype(np.float64))
    g, t = D.test_windows(w, D.split_slice(dcfg, "test"), ecfg["sets"]["test"]["min_live"])
    k = w[:, t].sum(axis=0) - w[g, t]
    ds = xr.Dataset({"target": ("sample", g.astype(np.int8)), "t": ("sample", t.astype(np.int32)), "n_live": ("sample", k.astype(np.int8))},
                    coords={"station": stations}, attrs={"region": region, "meta": json.dumps(io.run_meta({"data": dcfg, "eval": ecfg}, []))})
    io.write_netcdf(ds, C.test_set(region))
    print(f"{region}: {len(t):,} test windows -> {C.test_set(region)}")


if __name__ == "__main__":
    main()
