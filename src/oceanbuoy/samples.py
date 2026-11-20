from __future__ import annotations

import numpy as np
import xarray as xr

from . import io, rng as R, sensor_groups as GR


CHUNK = 500_000


def neighbour_count(n_stations: int, scfg: dict) -> int:
    n_nb = n_stations - 1
    assert n_nb <= scfg["stream"]["max_neighbors"], "nearest-neighbour selection is not implemented"
    return n_nb


def eligible(observed: np.ndarray, scfg: dict) -> np.ndarray:
    live_total = observed.sum(axis=0)
    return observed & ((live_total - 1)[None, :] >= scfg["stream"]["min_live"])


def draw_targets_times(ok: np.ndarray, t_lo: int, t_hi: int, n: int,
                       g: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    S = ok.shape[0]
    frac = ok[:, t_lo:t_hi].mean()
    tg, tt, have = [], [], 0
    while have < n:
        m = int((n - have) / frac * 1.1) + 1000
        cand_g = g.integers(0, S, size=m)
        cand_t = g.integers(t_lo, t_hi, size=m)
        keep = ok[cand_g, cand_t]
        tg.append(cand_g[keep]); tt.append(cand_t[keep]); have += int(keep.sum())
    return np.concatenate(tg)[:n], np.concatenate(tt)[:n]


def draw_keep(target: np.ndarray, t: np.ndarray, observed: np.ndarray, drop_max: int,
              g: np.random.Generator) -> np.ndarray:
    N, S = len(target), observed.shape[0]
    keep = np.zeros((N, S), bool)
    u = g.random(N)
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        rows = np.arange(b - a)
        live = observed[:, t[a:b]].T.copy()
        live[rows, target[a:b]] = False
        n_live = live.sum(axis=1)
        assert n_live.min() >= 1, "a sample without a live neighbour reached draw_keep"
        n_drop = np.floor(u[a:b] * (np.minimum(drop_max, n_live - 1) + 1)).astype(int)
        keys = g.random((b - a, S))
        keys[~live] = np.inf
        rank = np.argsort(np.argsort(keys, axis=1), axis=1)
        dropped = live & (rank < n_drop[:, None])
        keep[a:b] = ~dropped
        keep[a + rows, target[a:b]] = False
    return keep


def pack_bits(keep: np.ndarray) -> np.ndarray:
    assert keep.shape[1] <= 16
    return (keep.astype(np.uint16) << np.arange(keep.shape[1], dtype=np.uint16)).sum(axis=1).astype(np.uint16)


def unpack_bits(bits: np.ndarray, n_stations: int) -> np.ndarray:
    return ((bits[:, None] >> np.arange(n_stations, dtype=np.uint16)) & 1).astype(bool)


def time_dtype(t_hi: int):
    return np.uint16 if t_hi <= np.iinfo(np.uint16).max + 1 else np.int32


def make_stream(observed: np.ndarray, scfg: dict, seed: int, t_lo: int, t_hi: int) -> dict:
    neighbour_count(observed.shape[0], scfg)
    target, t = draw_targets_times(eligible(observed, scfg), t_lo, t_hi, scfg["stream"]["n_samples"],
                                   R.derive(seed, "data"))
    keep = draw_keep(target, t, observed, scfg["stream"]["drop_max"], R.derive(seed, "mask"))
    return {"target": target.astype(np.int8), "t": t.astype(time_dtype(t_hi)), "keep_bits": pack_bits(keep)}


def make_val_set(observed: np.ndarray, scfg: dict, t_lo: int, t_hi: int) -> dict:
    neighbour_count(observed.shape[0], scfg)
    target, t = np.nonzero(eligible(observed, scfg)[:, t_lo:t_hi])
    t = t + t_lo
    keep = draw_keep(target, t, observed, scfg["stream"]["drop_max"],
                     R.derive(scfg["val_set"]["mask_seed"], "valmask"))
    return {"target": target.astype(np.int8), "t": t.astype(time_dtype(t_hi)), "keep_bits": pack_bits(keep)}


N_MET = 3


def unpack_tmet(bits: np.ndarray) -> np.ndarray:
    return ((bits[:, None] >> np.arange(N_MET, dtype=np.uint8)) & 1).astype(bool)


def load_samples(path, gcfg: dict, with_bits: bool = True) -> dict:
    ds = xr.open_dataset(path, engine="h5netcdf")
    S = ds.sizes["station"]
    out = {"target": ds["target"].values.astype(np.int64), "t": ds["t"].values.astype(np.int64),
           "keep": unpack_bits(ds["keep_bits"].values, S), "S": S}
    if with_bits:
        out["gmask"] = np.stack([unpack_bits(ds[f"goff_{g}"].values, S) for g in gcfg["group_order"]], axis=2)
        out["tmet"] = unpack_tmet(ds["tmet_bits"].values)
    ds.close()
    out["sha256"] = io.sha256_file(path)
    return out


def subset(s: dict, rows: np.ndarray) -> dict:
    out = {**s, "target": s["target"][rows], "t": s["t"][rows], "keep": s["keep"][rows]}
    for k in ("gmask", "tmet"):
        if s.get(k) is not None:
            out[k] = s[k][rows]
    return out


N_GROUP = 4


def draw_copy(target: np.ndarray, counts: np.ndarray, g: np.random.Generator) -> np.ndarray:
    tot = counts.sum(axis=1)
    assert (tot[np.unique(target)] > 0).all(), "a target buoy has an empty pool"
    cum = np.cumsum(counts, axis=1)
    u = g.random(len(target))
    j = np.minimum((u * tot[target]).astype(np.int64), tot[target] - 1)
    out = np.empty(len(target), np.uint8)
    for b in np.unique(target):
        m = target == b
        out[m] = np.searchsorted(cum[b], j[m], side="right")
    return out


def target_state_bits(target: np.ndarray, t: np.ndarray, alive: np.ndarray, counts: np.ndarray, gcfg: dict, seed: int, region_index: int,
            ) -> tuple[dict, np.ndarray]:
    n, S = len(t), alive.shape[0]
    keep = np.ones((n, S), bool)
    keep[np.arange(n), target] = False
    s = draw_copy(target, counts, R.derive(seed, "tcopy", region_index))
    actual = GR.pattern_code(alive[target, t][:, GR.met_index(gcfg)])
    e = (s & actual).astype(np.uint8)
    off = np.zeros((n, S, N_GROUP), bool)
    cnt = GR.input_counts(target, t, keep, off, alive)
    wi = gcfg["group_order"].index("wave")
    lv = alive[:, t, wi].T.copy()
    lv[np.arange(n), target] = False
    out = {"tpat_bits": s, "tmet_bits": e, "n_live": lv.sum(axis=1).astype(np.uint8), "n_withheld": np.zeros(n, np.uint8),
           "n_wave_masked": np.zeros(n, np.uint8), "n_tokens": cnt["n_tokens"]}
    for k, name in enumerate(gcfg["group_order"]):
        out[f"n_{name}_in"] = cnt["per_group"][:, k]
    assert np.array_equal(out["n_wave_in"].astype(int), out["n_live"].astype(int)), "with nothing withheld every live neighbour is an input"
    assert out["n_wave_in"].min() >= 1, "a sample without a live neighbour"
    return out, keep
