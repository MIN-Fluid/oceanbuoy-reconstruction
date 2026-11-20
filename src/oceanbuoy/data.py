from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C

LOW_M, MIN_RUN_H, NB_MIN_M = 0.1, 12, 0.5


def open_obs(path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return ds["obs"].values, [str(s) for s in ds["station"].values], ds["time"].values, ds["lat"].values, ds["lon"].values


def split_slice(cfg: dict, split: str) -> slice:
    s = cfg["splits"][split]
    return slice(s["idx_start"], s["idx_end"] + 1)


def obs_from_aligned(region: str, dcfg: dict) -> xr.Dataset:
    src = C.cached("socal12") if region == C.TWELVE else dcfg["sources"][region]
    z = np.load(C.ROOT / src["aligned"], allow_pickle=False)
    meta = json.loads((C.ROOT / src["stations"]).read_text())
    data, stations = z["data"], [str(s) for s in z["stations"]]
    assert [str(v) for v in z["variables"]] == dcfg["variables"], "variable order differs from config"
    assert data.dtype == np.float32 and np.array_equal(z["unknown_mask"], np.isnan(data))
    ts = z["timestamps"].astype(np.int64)
    return xr.Dataset(
        {"obs": (("station", "time", "variable"), data)},
        coords={"station": stations,
                "time": pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).values.astype("datetime64[ns]"),
                "variable": dcfg["variables"],
                "units": ("variable", [dcfg["units"][v] for v in dcfg["variables"]]),
                "lat": ("station", np.array([meta[s]["lat"] for s in stations], np.float64)),
                "lon": ("station", np.array([meta[s]["lon"] for s in stations], np.float64)),
                "depth_m": ("station", np.array([meta[s]["depth_m"] for s in stations], np.float64))},
        attrs={"region": region, "missing": "NaN where the station did not report", "time_zone": "UTC"})


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def wave_fault_runs(hs: np.ndarray, dpd: np.ndarray, g: int) -> list[tuple[int, int, float]]:
    out = []
    for s, n in runs(hs[g] < LOW_M):
        e = s + n
        if n < MIN_RUN_H or np.isfinite(dpd[g, s:e]).any():
            continue
        nb = float(np.nanmedian(np.delete(hs, g, axis=0)[:, s:e]))
        if nb >= NB_MIN_M:
            out.append((s, n, nb))
    return out


def regional_norm(obs: np.ndarray, cfg: dict) -> dict:
    block = obs[:, split_slice(cfg, "train"), :].astype(np.float64)
    flat = block.reshape(-1, block.shape[-1])
    mean, std = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0)
    assert np.all(np.isfinite(mean)) and np.all(std > 0), "pooled train std must be > 0"
    n = np.isfinite(flat).sum(axis=0)
    idx = {v: i for i, v in enumerate(cfg["variables"])}
    return {v: {"mean": float(mean[idx[v]]), "std": float(std[idx[v]]), "n_obs_train": int(n[idx[v]])}
            for v in cfg["norm_variables"]}


def test_windows(w: np.ndarray, te: slice, min_live: int) -> tuple[np.ndarray, np.ndarray]:
    ok = w & (w.sum(axis=0)[None, :] - 1 >= min_live)
    ok[:, :te.start] = False
    ok[:, te.stop:] = False
    return np.nonzero(ok)
