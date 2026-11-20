from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import xarray as xr


REGIONS = ("socal", "gom", "pnw")
BLOCK_H = 240
BOOT_PURPOSE = 6


def load_score_config(root: Path) -> dict:
    s = json.loads((root / "configs" / "eval.json").read_text())
    return {"n_boot": s["score"]["n_boot"], "ci": s["score"]["ci"], "boot_seed": s["score"]["boot_seed"],
            "min_group_n": s["sets"]["strata"]["min_stratum_n"]}


def read_predictions(path: Path) -> dict:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        out = {"truth": ds["truth"].values.astype(np.float64), "target": ds["target"].values.astype(np.int64),
               "t": ds["t"].values.astype(np.int64), "pred": ds["pred"].values.astype(np.float64)}
        if "n_live" in ds:
            out["n_live"] = ds["n_live"].values.astype(int)
        if "seed" in ds.coords:
            out["seeds"] = [int(s) for s in ds["seed"].values]
    return out


def pattern_weights(pi: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    return [pi[target, p] for p in range(8)]


def weighted_sq(pred: np.ndarray, truth: np.ndarray, W: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ens, members = 0.0, 0.0
    for p in range(8):
        w = np.asarray(W[p], float)
        ens = ens + w * (pred[p].mean(axis=1) - truth) ** 2
        members = members + w[:, None] * (pred[p] - truth[:, None]) ** 2
    return ens, members


def outage_type_sq(pred: np.ndarray, truth: np.ndarray, W: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = [pred[p].mean(axis=1) - truth for p in range(8)]
    den = sum(np.broadcast_to(np.asarray(W[p], float), E[p].shape[:1]) for p in range(1, 8))
    num = sum(np.asarray(W[p], float) * E[p] ** 2 for p in range(1, 8))
    ok = den > 0
    return E[0] ** 2, np.where(ok, num / np.where(ok, den, 1.0), np.nan), ok


def group_bins(n_live: np.ndarray, min_n: int) -> list[tuple[int, int]]:
    vals, cnt = np.unique(n_live, return_counts=True)
    bins, pooled_hi = [], None
    for v, c in zip(vals, cnt):
        if c < min_n and not bins:
            pooled_hi = int(v)
            continue
        bins.append((int(v), int(v)))
    if pooled_hi is not None:
        bins.insert(0, (int(vals.min()), pooled_hi))
    return bins


class Bootstrap:

    def __init__(self, t: np.ndarray, truth: np.ndarray, region: str, cfg: dict):
        _, idx = np.unique(np.asarray(t) // BLOCK_H, return_inverse=True)
        self.bidx, self.n_blocks = idx.ravel(), int(idx.max()) + 1
        self.truth, self.cfg = truth, cfg
        g = np.random.default_rng(np.random.SeedSequence([int(cfg["boot_seed"]), BOOT_PURPOSE, REGIONS.index(region), BLOCK_H]))
        picks = g.integers(0, self.n_blocks, size=(cfg["n_boot"], self.n_blocks))
        flat = (picks + self.n_blocks * np.arange(cfg["n_boot"])[:, None]).ravel()
        self.counts = np.bincount(flat, minlength=cfg["n_boot"] * self.n_blocks).reshape(cfg["n_boot"], self.n_blocks)
        self.n = np.bincount(self.bidx, minlength=self.n_blocks).astype(np.float64)

    def error(self, sq: np.ndarray) -> np.ndarray:
        return (self.truth + np.sqrt(sq)) - self.truth

    def draws(self, sq: np.ndarray) -> np.ndarray:
        sse = np.bincount(self.bidx, weights=self.error(sq) ** 2, minlength=self.n_blocks)
        return np.sqrt((self.counts @ sse) / (self.counts @ self.n))

    def interval(self, draws: np.ndarray) -> list[float]:
        lo, hi = np.percentile(draws, self.cfg["ci"])
        return [float(lo), float(hi)]


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(e, np.float64) ** 2)))


def score(boot: Bootstrap, sq: dict[str, np.ndarray], pairs: list[tuple[str, str]], member_sq: dict | None = None) -> dict:
    draws = {m: boot.draws(v) for m, v in sq.items()}
    methods = {}
    for m, v in sq.items():
        e = np.sqrt(v)
        row = {"rmse_m": rmse(boot.error(v)), "ci95_m": boot.interval(draws[m]),
               "max_error_m": float(e.max()), "p95_error_m": float(np.percentile(e, 95)), "p99_error_m": float(np.percentile(e, 99))}
        if member_sq is not None and m in member_sq:
            y = boot.truth
            pred = y[:, None] + np.sqrt(member_sq[m])
            row["member_rmse_m"] = [rmse(pred[:, k] - y) for k in range(pred.shape[1])]
        methods[m] = row
    diffs = {f"{a}-{b}": {"d_rmse_m": methods[a]["rmse_m"] - methods[b]["rmse_m"], "ci95_m": boot.interval(draws[a] - draws[b])}
             for a, b in pairs}
    return {"n_samples": int(len(boot.truth)), "n_blocks": boot.n_blocks, "methods": methods, "differences": diffs}
