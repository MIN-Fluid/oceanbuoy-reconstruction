from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr

from . import config as C, io, models as M, ridge as RR, sensor_groups as GR, train as T, views as V


def load_set(region: str) -> dict:
    with xr.open_dataset(C.test_set(region), engine="h5netcdf") as ds:
        S = ds.sizes["station"]
        tg, t = ds["target"].values.astype(np.int64), ds["t"].values.astype(np.int64)
        n_live = ds["n_live"].values
    keep = np.ones((len(t), S), bool)
    keep[np.arange(len(t)), tg] = False
    return {"target": tg, "t": t, "keep": keep, "S": S, "n_live": n_live}


def target_reported(env: dict, target: np.ndarray, t: np.ndarray) -> np.ndarray:
    alive = GR.group_alive(env["obs"], env["gcfg"], env["dcfg"]["variables"])
    return alive[target, t][:, GR.met_index(env["gcfg"])]


def predict_net(path: Path, env: dict, s: dict) -> np.ndarray:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    d_in = ck["state_dict"]["body.net.0.weight"].shape[1] if m["model"] == "mlp" else None
    model = M.build_model(m["model"], C.load("models"), m["seed"], d_in=d_in, target=m["target"] or 0)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    pcfg = C.load("eval")["predict"]
    inputs = T.inputs_fn(m["model"], env, m["target"])
    out = []
    with torch.no_grad():
        for a in range(0, len(s["t"]), pcfg["batch"]):
            x = T.to_device(inputs(s, a, min(a + pcfg["batch"], len(s["t"]))), pcfg["device"])
            out.append(model(*x).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def predict_patterns(ckpts: list[Path], env: dict, s: dict, actual: np.ndarray) -> np.ndarray:
    M.configure_torch(C.load("models"))
    out = np.full((8, len(s["t"]), len(ckpts)), np.nan)
    for p in range(8):
        sp = {**s, "tmet": GR.pattern_tmet(p, actual)}
        for j, ck in enumerate(ckpts):
            out[p, :, j] = predict_net(ck, env, sp)
    assert np.isfinite(out).all(), "a window has no prediction"
    return out


def predict_test(method: str, region: str, seeds: list[int], L: C.Layout) -> Path:
    M.configure_torch(C.load("models"))
    env = V.env(region)
    s = load_set(region)
    truth = env["obs"][:, :, env["dcfg"]["variables"].index("WVHT")].astype(np.float64)[s["target"], s["t"]]
    assert np.isfinite(truth).all()
    actual = target_reported(env, s["target"], s["t"])
    S = s["S"]
    if method in ("attention", "token_mlp"):
        P = predict_patterns([L.ckpt(method, region, seed) for seed in seeds], env, s, actual)
    else:
        P = np.full((8, len(truth), len(seeds)), np.nan)
        for k, seed in enumerate(seeds):
            if method == "ridge":
                with xr.open_dataset(L.ridge(region, seed), engine="h5netcdf") as c:
                    coef = [{kk: c[kk].values[g] for kk in ("w", "mu", "sd", "intercept")} for g in range(S)]
            for g in range(S):
                rows = np.nonzero(s["target"] == g)[0]
                if len(rows) == 0:
                    continue
                for p in range(8):
                    sub = {"target": s["target"][rows], "t": s["t"][rows], "keep": s["keep"][rows], "S": S, "tmet": GR.pattern_tmet(p, actual[rows])}
                    if method == "mlp":
                        P[p, rows, k] = predict_net(L.ckpt(method, region, seed, g), env, sub)
                    else:
                        X = V.make_inputs("ridge", env["tab"], env["vcfg"], env["cm"], True, sub, 0, len(rows), g)[0]
                        P[p, rows, k] = RR.predict(coef[g], X)
    assert np.isfinite(P).all(), "a window has no prediction"
    ds = xr.Dataset({"target": ("window", s["target"].astype(np.int8)), "t": ("window", s["t"].astype(np.int32)), "truth": ("window", truth),
                     "n_live": ("window", s["n_live"].astype(np.int8)), "pred": (("pattern", "window", "seed"), P)},
                    coords={"pattern": np.arange(8), "seed": seeds, "station": env["stations"]}, attrs={"method": method, "region": region})
    path = C.pred(method, region)
    io.write_netcdf(ds, path)
    return path
