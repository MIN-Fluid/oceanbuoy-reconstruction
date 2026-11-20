from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import xarray as xr


class RecVal:

    def __init__(self, target, t, S, item_row, item_code, item_w, y_rows, pi):
        self.target, self.t, self.S, self.pi = target, t, S, pi
        self.item_row, self.item_code, self.item_w = item_row, item_code, item_w
        self.y_rows = np.asarray(y_rows, np.float64)
        self.n_rows = len(target)
        keep = np.ones((self.n_rows, S), bool)
        keep[np.arange(self.n_rows), target] = False
        tmet = ((item_code[:, None] >> np.arange(3, dtype=np.uint8)) & 1).astype(bool)
        self.s = {"target": target[item_row], "t": t[item_row], "keep": keep[item_row], "S": S, "tmet": tmet}
        self.y_items = self.y_rows[item_row]

    @property
    def n_items(self) -> int:
        return len(self.item_row)

    def score(self, p: np.ndarray) -> float:
        d = np.asarray(p, np.float64) - self.y_items
        return float(np.sqrt(np.sum(self.item_w * d * d) / self.n_rows))

    def subset_target(self, g: int) -> "RecVal":
        rows = np.nonzero(self.target == g)[0]
        remap = -np.ones(self.n_rows, np.int64)
        remap[rows] = np.arange(len(rows))
        sel = np.nonzero(remap[self.item_row] >= 0)[0]
        return RecVal(self.target[rows], self.t[rows], self.S, remap[self.item_row[sel]], self.item_code[sel], self.item_w[sel],
                      self.y_rows[rows], self.pi)


def truth(obs: np.ndarray, target: np.ndarray, t: np.ndarray, dcfg: dict) -> np.ndarray:
    return obs[target, t, dcfg["variables"].index("WVHT")].astype(np.float32).astype(np.float64)


def load_recval(path: Path, obs: np.ndarray, dcfg: dict) -> RecVal:
    ds = xr.open_dataset(path, engine="h5netcdf")
    tg, t = ds["target"].values.astype(np.int64), ds["t"].values.astype(np.int64)
    ir, ic, iw = ds["item_row"].values.astype(np.int64), ds["item_code"].values.astype(np.uint8), ds["item_w"].values
    S, pi = ds.sizes["station"], np.asarray(json.loads(ds.attrs["pi"]), float)
    ds.close()
    y = truth(obs, tg, t, dcfg)
    assert np.isfinite(y).all(), "a validation label is NaN"
    return RecVal(tg, t, S, ir, ic, iw, y, pi)


def build_items_pib(target: np.ndarray, actual: np.ndarray, pib: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(target)
    W = np.zeros((n, 8))
    rows = np.arange(n)
    for p in range(8):
        np.add.at(W, (rows, p & actual.astype(np.int64)), pib[target, p])
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12, rtol=0), "item weights of a row must sum to 1"
    rr, cc = np.nonzero(W)
    return rr.astype(np.int64), cc.astype(np.uint8), W[rr, cc]
