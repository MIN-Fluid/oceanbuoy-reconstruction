from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import config as C, data as D, interp as I, io, predict as PR, views as V

RUNGS = ("none", "ft_emb", "ft_enc", "ft", "retrain")
ADDED = {"46239": 10, "46047": 11}
ORIG = 10


def rung_ckpts(rung: str, L: C.Layout, seeds: list[int]) -> list[Path]:
    if rung == "none":
        return [L.ckpt("attention", "socal", x) for x in seeds]
    return [L.expansion(rung, x) for x in seeds]


def _write(path: Path, w: dict, pred: np.ndarray, seeds: list[int] | None, attrs: dict, n_live: np.ndarray | None = None) -> Path:
    v = {"target": ("window", w["tg"].astype(np.int8)), "t": ("window", w["t"].astype(np.int32)), "truth": ("window", w["truth"])}
    if n_live is not None:
        v["n_live"] = ("window", n_live.astype(np.int8))
    if pred.ndim == 3:
        ds = xr.Dataset({**v, "pred": (("pattern", "window", "seed"), pred)}, coords={"pattern": np.arange(8), "seed": seeds}, attrs=attrs)
    else:
        ds = xr.Dataset({**v, "pred": ("window", pred)}, attrs=attrs)
    io.write_netcdf(ds, path)
    return path


def original_buoys(seeds: list[int], L: C.Layout) -> list[Path]:
    env10, env12 = V.env("socal"), V.env(C.TWELVE)
    s0 = PR.load_set("socal")
    tg, t = s0["target"], s0["t"]
    hs = env10["obs"][:, :, env10["dcfg"]["variables"].index("WVHT")].astype(np.float64)
    w = {"tg": tg, "t": t, "truth": hs[tg, t]}
    keep = np.ones((len(t), 12), bool)
    keep[np.arange(len(t)), tg] = False
    s = {"target": tg, "t": t, "keep": keep, "S": 12}
    actual = PR.target_reported(env10, tg, t)
    out = []
    for rung in RUNGS:
        P = PR.predict_patterns(rung_ckpts(rung, L, seeds), env12, s, actual)
        out.append(_write(C.pred("expansion", rung), w, P, seeds, {"rung": rung, "region": "socal", "inputs": "12-buoy record"}))
    return out


def added_buoys(seeds: list[int], L: C.Layout) -> list[Path]:
    env12 = V.env(C.TWELVE)
    icfg = C.load("interp")
    kc = icfg["kriging"]
    par = json.loads(C.interp_params(C.TWELVE).read_text())
    _, _, _, lat, lon = D.open_obs(C.obs(C.TWELVE))
    xy = I.planar_km(lat, lon, C.base_region(C.TWELVE), C.load("views"))
    hs = env12["obs"][:, :, env12["dcfg"]["variables"].index("WVHT")].astype(np.float64)
    te = D.split_slice(env12["dcfg"], "test")
    ts = np.arange(te.start, te.stop)
    n_live_all = np.isfinite(hs[:ORIG, ts]).sum(axis=0)
    out = []
    for b, g in ADDED.items():
        sel = np.isfinite(hs[g, ts]) & (n_live_all >= 2)
        t = ts[sel].astype(np.int64)
        tg = np.full(len(t), g, np.int64)
        w, n_live = {"tg": tg, "t": t, "truth": hs[g, t]}, n_live_all[sel].astype(int)
        keep = np.ones((len(t), 12), bool)
        keep[np.arange(len(t)), tg] = False
        s = {"target": tg, "t": t, "keep": keep, "S": 12}
        actual = PR.target_reported(env12, tg, t)
        for m in RUNGS:
            P = PR.predict_patterns(rung_ckpts(m, L, seeds), env12, s, actual)
            out.append(_write(C.pred("added_buoys", f"{m}_{b}"), w, P, seeds, {"method": m, "buoy": b}, n_live))
        avail, vals = I.inputs(hs, tg, t, keep)
        assert (avail.sum(axis=1) >= kc["min_neighbours"]).all()
        for m, pred in I.predict(par, xy, kc, icfg, tg, avail, vals).items():
            assert np.isfinite(pred).all(), f"{m}: a window has no estimate"
            out.append(_write(C.pred("added_buoys", f"{m}_{b}"), w, pred, None, {"method": m, "buoy": b}, n_live))
    return out
