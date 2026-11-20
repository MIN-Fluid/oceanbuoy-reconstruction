from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, data as D, interp as I, io, samples as SA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=C.ALL)
    region = ap.parse_args().region
    dcfg, scfg, icfg, sr = C.load("data"), C.load("sampling"), C.load("interp"), C.load("design")["interp"]["idw_search"]
    powers = [float(x) for x in np.arange(sr["low"], sr["high"] + sr["step"] / 2, sr["step"])]
    obs, stations, _, lat, lon = D.open_obs(C.obs(region))
    hs = obs[:, :, dcfg["variables"].index("WVHT")].astype(np.float64)
    xy = I.planar_km(lat, lon, C.base_region(region), C.load("views"))
    lo, hi = D.split_slice(dcfg, "train").start, D.split_slice(dcfg, "val").stop
    cov = I.pair_covariances(hs[:, lo:hi], icfg["kriging"]["min_overlap_hours"])
    tg, t = np.nonzero(SA.eligible(np.isfinite(hs), scfg)[:, lo:hi])
    t = t + lo
    keep = np.ones((len(t), hs.shape[0]), bool)
    keep[np.arange(len(t)), tg] = False
    avail, vals = I.inputs(hs, tg, t, keep)
    y = hs[tg, t]
    D0 = I.pair_dist(xy)
    rm = [I.rmse(I.combine(I.idw_weights(D0, tg, avail, p, icfg["idw"]["min_distance_km"]), vals), y) for p in powers]
    idw = {"p_star": powers[int(np.argmin(rm))], "rmse_by_p": rm, "powers": powers}
    ok = I.fit_ok(D0, cov, icfg["kriging"], tg, avail, vals, y)
    ak = I.select_ak(xy, cov, icfg["kriging"], icfg["aniso"], tg, avail, vals, y, icfg["boundary_tolerance"])
    out = {"meta": io.run_meta({"interp": icfg, "data": dcfg}, [C.obs(region)]), "region": region, "stations": stations, "xy_km": xy.tolist(),
           "n_fit_rows": int(len(t)), "idw": idw, "ok": ok, "ak": ak}
    io.write_json(C.interp_params(region), out)
    print(f"{region}: IDW p* {idw['p_star']}, OK c0 {ok['c0']:.4f} L {ok['L_km']:.1f} km, AK ratio {ak['ratio']} angle {ak['angle_deg']}")


if __name__ == "__main__":
    main()
