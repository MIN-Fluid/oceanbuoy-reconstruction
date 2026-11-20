from __future__ import annotations

import numpy as np

from . import sensor_groups as GR


N_STATE = 8


def _fractions(counts: np.ndarray) -> list[float] | None:
    n = int(counts.sum())
    return None if n == 0 else (counts / n).tolist()


def pool_states(alive_wave: np.ndarray, rep: np.ndarray, met_code: np.ndarray, sl: slice) -> dict:
    r = rep[sl]
    idx = np.nonzero(r)[0]
    if len(idx) == 0:
        return {"span": None, "n": 0, "counts": [0] * N_STATE, "fractions": None}
    span = np.zeros(r.shape, bool)
    span[idx[0]:idx[-1] + 1] = True
    pool = span & ~alive_wave[sl]
    counts = np.bincount(met_code[sl][pool], minlength=N_STATE)
    first = sl.start + int(idx[0])
    return {"span": [first, sl.start + int(idx[-1])], "n": int(pool.sum()), "counts": counts.tolist(), "fractions": _fractions(counts)}


def record_states(obs: np.ndarray, gcfg: dict, variables: list[str], tr: slice, stations: list[str]) -> dict:
    alive = GR.group_alive(obs, gcfg, variables)
    wave = alive[:, :, gcfg["group_order"].index("wave")]
    rep = np.isfinite(obs).any(axis=2)
    code = GR.pattern_code(alive[:, :, GR.met_index(gcfg)])
    full = slice(0, obs.shape[1])
    train = [pool_states(wave[g], rep[g], code[g], tr) for g in range(len(stations))]
    whole = [pool_states(wave[g], rep[g], code[g], full) for g in range(len(stations))]
    return {"train_counts": np.array([p["counts"] for p in train], np.int64), "train_fractions": np.array([p["fractions"] for p in train], float),
            "all_fractions": np.array([p["fractions"] for p in whole], float), "all_n": [p["n"] for p in whole], "alive": alive}
