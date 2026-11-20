from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, data as D, interp as I, io, predict as PR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=C.REGIONS)
    region = ap.parse_args().region
    icfg = C.load("interp")
    s = PR.load_set(region)
    obs, stations, _, lat, lon = D.open_obs(C.obs(region))
    hs = obs[:, :, C.load("data")["variables"].index("WVHT")].astype(np.float64)
    xy = I.planar_km(lat, lon, region, C.load("views"))
    par = json.loads(C.interp_params(region).read_text())
    assert par["stations"] == stations
    avail, vals = I.inputs(hs, s["target"], s["t"], s["keep"])
    n_live = avail.sum(axis=1)
    for m, pred in I.predict(par, xy, icfg["kriging"], icfg, s["target"], avail, vals).items():
        answered = np.isfinite(pred)
        assert answered[n_live >= (1 if m == "idw" else icfg["kriging"]["min_neighbours"])].all(), f"{m}: a window has no prediction"
        ds = xr.Dataset({"target": ("sample", s["target"].astype(np.int8)), "t": ("sample", s["t"].astype(np.int32)),
                         "truth": ("sample", hs[s["target"], s["t"]]), "n_live": ("sample", n_live.astype(np.int8)), "pred": ("sample", pred)},
                        coords={"station": stations}, attrs={"method": m, "region": region})
        io.write_netcdf(ds, C.pred(m, region))
        print(f"{m} {region}: {int(answered.sum()):,}/{len(pred):,} windows -> {C.pred(m, region)}")


if __name__ == "__main__":
    main()
