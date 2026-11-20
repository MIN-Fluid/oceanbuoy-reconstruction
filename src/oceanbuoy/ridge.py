from __future__ import annotations

import numpy as np
import json
from pathlib import Path
import xarray as xr

from . import config as C, io, samples as SA, train as T, views as V


class AlphaAtBoundary(RuntimeError):
    pass


def predict(coef: dict, X: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float64) - coef["mu"]) / coef["sd"]) @ coef["w"] + coef["intercept"]


def _chunks(design, s: dict, chunk: int):
    for a in range(0, len(s["t"]), chunk):
        yield design(s, a, min(a + chunk, len(s["t"]))).astype(np.float64)


def fit_target(env: dict, rcfg: dict, g: int, train_s: dict, val, design) -> dict:
    chunk = rcfg["chunk_rows"]
    y = V.truth(env["obs"], train_s["target"], train_s["t"], env["dcfg"]).astype(np.float64)
    n, s1, s2 = 0, None, None
    for X in _chunks(design, train_s, chunk):
        s1 = X.sum(0) if s1 is None else s1 + X.sum(0)
        s2 = (X ** 2).sum(0) if s2 is None else s2 + (X ** 2).sum(0)
        n += len(X)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0.0))
    sd[sd < 1e-12] = 1.0
    ybar = y.mean()
    d = len(mu)
    G, bvec, a = np.zeros((d, d)), np.zeros(d), 0
    for X in _chunks(design, train_s, chunk):
        Z = (X - mu) / sd
        G += Z.T @ Z
        bvec += Z.T @ (y[a:a + len(X)] - ybar)
        a += len(X)
    Zv = np.concatenate([(X - mu) / sd for X in _chunks(design, val.s, chunk)])
    alphas = rcfg["alphas"]
    ws, crit = [], []
    for al in alphas:
        w = np.linalg.solve(G + al * np.eye(d), bvec)
        ws.append(w)
        crit.append(val.score(Zv @ w + ybar))
    k = int(np.argmin(crit))
    edge_flat = False
    if k in (0, len(alphas) - 1):
        nb = 1 if k == 0 else len(alphas) - 2
        gain = (crit[nb] - crit[k]) / crit[nb]
        if gain > rcfg["boundary_tolerance"]:
            raise AlphaAtBoundary(f"target {g}: alpha {alphas[k]} is at the edge of the grid and beats its neighbour by {100 * gain:.3f}% ({crit})")
        edge_flat = True
    return {"w": ws[k], "mu": mu, "sd": sd, "intercept": ybar, "alpha": alphas[k], "edge_flat": edge_flat, "val_by_alpha": crit,
            "n_train": n, "n_val": int(val.n_rows)}


def train_ridge(region: str, seed: int, L: C.Layout, log=print) -> Path:
    env = V.env(region)
    train_s, val_all, files = T.load_inputs(region, seed, L, env)
    rcfg = {**C.load("train")["ridge"], "alphas": C.load("pipeline")["ridge"]["alphas"]}
    fits = []
    for g in range(env["tab"]["S"]):
        tr = SA.subset(train_s, np.nonzero(train_s["target"] == g)[0])
        design = lambda s, i, j, g=g: V.make_inputs("ridge", env["tab"], env["vcfg"], env["cm"], True, s, i, j, g)[0]
        f = fit_target(env, rcfg, g, tr, val_all.subset_target(g), design)
        fits.append(f)
        log(f"  target {g}: alpha {f['alpha']}{' (flat edge)' if f['edge_flat'] else ''}, best criterion {min(f['val_by_alpha']):.4f}, "
            f"{f['n_train']:,} train rows, {f['n_val']:,} val rows")
    ds = xr.Dataset(
        {"w": (("target", "column"), np.stack([f["w"] for f in fits])), "mu": (("target", "column"), np.stack([f["mu"] for f in fits])),
         "sd": (("target", "column"), np.stack([f["sd"] for f in fits])), "intercept": ("target", np.array([f["intercept"] for f in fits])),
         "alpha": ("target", np.array([f["alpha"] for f in fits])), "edge_flat": ("target", np.array([f["edge_flat"] for f in fits])),
         "val_by_alpha": (("target", "alpha_candidate"), np.array([f["val_by_alpha"] for f in fits])),
         "n_train": ("target", np.array([f["n_train"] for f in fits])), "n_val": ("target", np.array([f["n_val"] for f in fits]))},
        coords={"alpha_candidate": rcfg["alphas"]},
        attrs={"region": region, "seed": seed, "solver": rcfg["solver"], "files_sha256": json.dumps({k: io.sha256_file(v) for k, v in files.items()})})
    out = L.ridge(region, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    io.write_netcdf(ds, out)
    return out
