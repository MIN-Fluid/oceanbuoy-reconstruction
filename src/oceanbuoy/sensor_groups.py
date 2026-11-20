from __future__ import annotations

import numpy as np


def column_groups(gcfg: dict, vcfg: dict) -> dict:
    names = gcfg["group_order"]
    of = {v: k for k, g in enumerate(names) for v in gcfg["groups"][g]}
    val = np.array([of[c[:4] if c.startswith("WDIR") else c] for c in vcfg["value_columns"]])
    flag = np.array([of[c] for c in vcfg["flag_columns"]])
    assert len(val) == 8 and len(flag) == 7
    assert set(val) == set(flag) == set(range(len(names))), "the groups must partition the token columns"
    return {"val": val, "flag": flag, "names": names}


def group_alive(obs: np.ndarray, gcfg: dict, variables: list[str]) -> np.ndarray:
    fin = np.isfinite(obs)
    return np.stack([fin[:, :, [variables.index(v) for v in gcfg["groups"][g]]].any(axis=2)
                     for g in gcfg["group_order"]], axis=2)


def met_index(gcfg: dict) -> list[int]:
    return [gcfg["group_order"].index(g) for g in gcfg["met_groups"]]


def pattern_code(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint8) << np.arange(bits.shape[-1], dtype=np.uint8)).sum(axis=-1).astype(np.uint8)


def input_counts(target: np.ndarray, t: np.ndarray, keep: np.ndarray, off: np.ndarray, alive: np.ndarray) -> dict:
    rows = np.arange(len(t))
    a = alive[:, t].transpose(1, 0, 2).copy()
    a[rows, target] = False
    on = a & ~off & keep[:, :, None]
    return {"per_group": on.sum(axis=1).astype(np.uint8),
            "n_tokens": on.any(axis=2).sum(axis=1).astype(np.uint8)}


def pattern_tmet(p: int, actual: np.ndarray) -> np.ndarray:
    assert 0 <= p <= 7 and actual.ndim == 2 and actual.shape[1] == 3
    bits = ((p >> np.arange(3)) & 1).astype(bool)
    return actual & bits[None, :]
