from __future__ import annotations

import numpy as np


class ParamAtBoundary(RuntimeError):
    pass


def planar_km(lat: np.ndarray, lon: np.ndarray, region: str, vcfg: dict) -> np.ndarray:
    ref_lat, ref_lon = vcfg["coordinate_reference"][region]
    east = (lon - ref_lon) * 111.320 * np.cos(np.radians(ref_lat))
    north = (lat - ref_lat) * 110.574
    return np.stack([east, north], axis=1).astype(np.float64)


def pair_dist(xy: np.ndarray, ratio: float = 1.0, angle_deg: float = 0.0) -> np.ndarray:
    th = np.radians(angle_deg)
    rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    z = xy @ (np.diag([1.0, 1.0 / ratio]) @ rot).T
    return np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)


def inputs(hs: np.ndarray, target: np.ndarray, t: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = hs[:, t].T.astype(np.float64)
    avail = np.isfinite(v) & keep
    avail[np.arange(len(t)), target] = False
    return avail, np.where(avail, v, 0.0)


def combine(W: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return (W * vals).sum(axis=1)


def idw_weights(D: np.ndarray, target: np.ndarray, avail: np.ndarray, p: float, dmin: float) -> np.ndarray:
    raw = avail / np.maximum(D[target], dmin) ** p
    s = raw.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, raw / s, np.nan)


def pair_covariances(hs_train: np.ndarray, min_overlap: int) -> np.ndarray:
    S = hs_train.shape[0]
    fin = np.isfinite(hs_train)
    C = np.full((S, S), np.nan)
    for i in range(S):
        for j in range(S):
            m = fin[i] & fin[j]
            if m.sum() >= min_overlap:
                C[i, j] = np.cov(hs_train[i, m], hs_train[j, m])[0, 1]
    return C


def kriging_solve(D: np.ndarray, c0: float, L: float, nugget: float, g: int, idx: np.ndarray) -> np.ndarray:
    n = len(idx)
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = c0 * np.exp(-D[np.ix_(idx, idx)] / L) + nugget * np.eye(n)
    M[n, :n] = M[:n, n] = 1.0
    r = np.empty(n + 1)
    r[:n] = c0 * np.exp(-D[g, idx] / L)
    r[n] = 1.0
    return np.linalg.solve(M, r)[:n]


def kriging_weights(D: np.ndarray, c0: float, L: float, kcfg: dict, target: np.ndarray, avail: np.ndarray) -> np.ndarray:
    R, S = avail.shape
    key = target.astype(np.int64) * (1 << S) + (avail.astype(np.int64) << np.arange(S)).sum(axis=1)
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    Wu = np.full((len(uniq), S), np.nan)
    for u, r in enumerate(first):
        idx = np.nonzero(avail[r])[0]
        if len(idx) < kcfg["min_neighbours"]:
            continue
        Wu[u] = 0.0
        Wu[u, idx] = kriging_solve(D, c0, L, kcfg["nugget_m2"], int(target[r]), idx)
    return Wu[inv.ravel()]


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def edge_check(rmses: list[float], k: int, ends: tuple[int, ...], tol: float, what: str) -> bool:
    if k not in ends:
        return False
    nb = k + 1 if k == 0 else k - 1
    gain = (rmses[nb] - rmses[k]) / rmses[nb]
    if gain > tol:
        raise ParamAtBoundary(f"{what}: candidate {k} is at the edge of the grid and beats its neighbour by "
                              f"{100 * gain:.3f}% (val RMSE {rmses})")
    return True


def fit_cov(D: np.ndarray, C: np.ndarray, kcfg: dict) -> dict | None:
    i, j = np.triu_indices(D.shape[0], 1)
    d, c = D[i, j], C[i, j]
    var = np.diag(C)
    if not (np.isfinite(var).all() and (var > 0).all()):
        return None
    pos = np.isfinite(c) & (c > 0)
    if pos.sum() < kcfg["min_positive_pairs"]:
        return None
    rho = c / np.sqrt(var[i] * var[j])
    A = np.stack([np.ones(int(pos.sum())), -d[pos]], axis=1)
    log_rho0, inv_L = np.linalg.lstsq(A, np.log(rho[pos]), rcond=None)[0]
    if inv_L <= 0:
        return None
    rho0, pooled = float(np.exp(log_rho0)), float(var.mean())
    return {"c0": rho0 * pooled, "L_km": float(1.0 / inv_L), "rho0": rho0, "mean_variance_m2": pooled, "inv_L_per_km": float(inv_L),
            "n_pairs": int(len(d)), "n_positive_pairs": int(pos.sum()), "rho_range": [float(rho[pos].min()), float(rho[pos].max())]}


def fit_ok(D: np.ndarray, C: np.ndarray, kcfg: dict, target: np.ndarray, avail: np.ndarray, vals: np.ndarray, y: np.ndarray) -> dict:
    det = fit_cov(D, C, kcfg)
    if det is None:
        raise RuntimeError("ordinary kriging: the correlation fit failed (too few positive pairs or L <= 0)")
    pred = combine(kriging_weights(D, det["c0"], det["L_km"], kcfg, target, avail), vals)
    ok = np.isfinite(pred)
    return {"c0": det["c0"], "L_km": det["L_km"], "val_rmse": rmse(pred[ok], y[ok]), "n_val_answered": int(ok.sum()), "n_val": len(y), "fit": det}


def select_ak(xy: np.ndarray, C: np.ndarray, kcfg: dict, acfg: dict, target: np.ndarray, avail: np.ndarray, vals: np.ndarray, y: np.ndarray, tol: float) -> dict:
    answer = avail.sum(axis=1) >= kcfg["min_neighbours"]
    cands = []
    for ratio in acfg["ratios"]:
        for ang in (acfg["angles_deg"] if ratio < 1.0 else [0]):
            D = pair_dist(xy, ratio, ang)
            fit = fit_cov(D, C, kcfg)
            if fit is None:
                cands.append({"ratio": ratio, "angle_deg": ang, "c0": None, "L_km": None, "val_rmse": None})
                continue
            c0, L = fit["c0"], fit["L_km"]
            pred = combine(kriging_weights(D, c0, L, kcfg, target, avail), vals)
            assert np.array_equal(np.isfinite(pred), answer), "candidates must be scored on the same rows"
            cands.append({"ratio": ratio, "angle_deg": ang, "c0": c0, "L_km": L, "val_rmse": rmse(pred[answer], y[answer])})
    ok = [c for c in cands if c["val_rmse"] is not None]
    best = min(ok, key=lambda c: c["val_rmse"])
    by_ratio = [min((c["val_rmse"] for c in ok if c["ratio"] == r), default=np.inf) for r in acfg["ratios"]]
    flat = edge_check(by_ratio, acfg["ratios"].index(best["ratio"]), (len(by_ratio) - 1,), tol, "AK ratio")
    return {**best, "edge_flat": flat, "candidates": cands, "val_rmse_by_ratio": by_ratio, "n_val_answered": int(answer.sum()), "n_val": len(y)}


def predict(par: dict, xy: np.ndarray, kc: dict, icfg: dict, target: np.ndarray, avail: np.ndarray, vals: np.ndarray) -> dict:
    D0 = pair_dist(xy)
    return {"idw": combine(idw_weights(D0, target, avail, par["idw"]["p_star"], icfg["idw"]["min_distance_km"]), vals),
            "ok": combine(kriging_weights(D0, par["ok"]["c0"], par["ok"]["L_km"], kc, target, avail), vals),
            "ak": combine(kriging_weights(pair_dist(xy, par["ak"]["ratio"], par["ak"]["angle_deg"]), par["ak"]["c0"], par["ak"]["L_km"], kc,
                                          target, avail), vals)}
