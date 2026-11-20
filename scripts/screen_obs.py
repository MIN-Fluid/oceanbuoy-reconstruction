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
    ap.add_argument("--region", required=True, choices=C.ALL)
    region = ap.parse_args().region
    dcfg, qcfg = C.load("data"), C.load("sensor_groups")["qc"]
    tr = D.split_slice(dcfg, "train")
    with xr.open_dataset(C.obs_unscreened(region), engine="h5netcdf") as ds:
        ds = ds.load()
    obs = ds["obs"].values.copy()
    stations = [str(s) for s in ds["station"].values]
    hours = ds["time"].values.astype("datetime64[h]")
    v = dcfg["variables"]
    if region == C.TWELVE:
        hs, dpd = obs[:, :, v.index("WVHT")].astype(np.float64), obs[:, :, v.index("DPD")].astype(np.float64)
        added = [stations.index(b) for b in C.cached("socal12")["added"]]
        spans = [(g, s, s + n) for g in added for s, n, _ in D.wave_fault_runs(hs, dpd, g) if tr.start <= s and s + n <= tr.stop]
    else:
        spans = []
        for r in json.loads(C.suspect_wave_runs().read_text())["regions"][region]["runs"]:
            a = int(np.nonzero(hours == np.datetime64(r["from"], "h"))[0][0])
            b = int(np.nonzero(hours == np.datetime64(r["to"], "h"))[0][0]) + 1
            assert tr.start <= a and b <= tr.stop, f"a wave-fault run outside the training years: {r}"
            spans.append((stations.index(r["station"]), a, b))
    for g, a, b in spans:
        for k in (v.index(x) for x in qcfg["variables"]):
            obs[g, a:b, k] = np.nan
    new = ds.copy(data={"obs": obs})
    new.attrs = {**{k: x for k, x in ds.attrs.items() if k != "meta"}, "qc_runs": json.dumps([[stations[g], a, b] for g, a, b in spans]),
                 "meta": json.dumps(io.run_meta({"data": dcfg, "qc": qcfg}, [C.obs_unscreened(region)]))}
    io.write_netcdf(new, C.obs(region))
    print(f"{region}: {len(spans)} wave-fault runs removed -> {C.obs(region)}")


if __name__ == "__main__":
    main()
