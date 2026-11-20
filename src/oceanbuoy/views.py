from __future__ import annotations

import json
import numpy as np

from . import config as C, data as D, sensor_groups as GR


N_VAL, N_FLAG, N_CAL, N_GEO = 8, 7, 4, 2
F = N_VAL + N_FLAG + N_CAL + N_GEO + 1
C_VAL, C_FLAG, C_CAL, C_GEO, C_TGT = 0, 8, 15, 19, 21


def station_table(obs: np.ndarray, times_s: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                  norm: dict, region: str, dcfg: dict, vcfg: dict) -> dict:
    variables = dcfg["variables"]
    vi = {v: i for i, v in enumerate(variables)}
    S, T, _ = obs.shape
    flags = np.isfinite(obs)
    vals = np.zeros((S, T, N_VAL), np.float32)
    for k, v in enumerate(vcfg["value_columns"][:6]):
        c = norm[v]
        z = (obs[:, :, vi[v]].astype(np.float64) - c["mean"]) / c["std"]
        vals[:, :, k] = np.where(flags[:, :, vi[v]], z, 0.0)
    wd = np.radians(obs[:, :, vi["WDIR"]])
    vals[:, :, 6] = np.where(flags[:, :, vi["WDIR"]], np.sin(wd), 0.0)
    vals[:, :, 7] = np.where(flags[:, :, vi["WDIR"]], np.cos(wd), 0.0)
    day, year = vcfg["seconds_per_day"], vcfg["seconds_per_year"]
    tod, doy = (times_s % day) / day, (times_s % year) / year
    cal = np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
                    np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)], axis=1).astype(np.float32)
    ref_lat, ref_lon = vcfg["coordinate_reference"][region]
    scale = vcfg["distance_scale_km"]
    east = (lon - ref_lon) * 111.320 * np.cos(np.radians(ref_lat)) / scale
    north = (lat - ref_lat) * 110.574 / scale
    geo = np.stack([east, north], axis=1).astype(np.float32)
    flag_order = [vi[v] for v in vcfg["flag_columns"]]
    return {"vals": vals, "flags": flags[:, :, flag_order], "cal": cal, "geo": geo,
            "S": S, "variables": variables}


def width_masks(width: str, vcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    keep_vars = set(vcfg["widths"][width])
    val_m = np.array([(c[:4] if c.startswith("WDIR") else c) in keep_vars for c in vcfg["value_columns"]])
    flag_m = np.array([c in keep_vars for c in vcfg["flag_columns"]])
    return val_m, flag_m


def _query(tab: dict, target: np.ndarray, t: np.ndarray) -> np.ndarray:
    q = np.zeros((len(t), F), np.float32)
    q[:, C_CAL:C_GEO] = tab["cal"][t]
    q[:, C_GEO:C_TGT] = tab["geo"][target]
    q[:, C_TGT] = 1.0
    return q


def truth(obs: np.ndarray, target: np.ndarray, t: np.ndarray, dcfg: dict) -> np.ndarray:
    return obs[target, t, dcfg["variables"].index("WVHT")].astype(np.float32)


WIDTH = "all"


def colmap(gcfg: dict, vcfg: dict, icfg: dict) -> dict:
    cg = GR.column_groups(gcfg, vcfg)
    mi = GR.met_index(gcfg)
    cg["met"] = [{"val": np.nonzero(cg["val"] == gi)[0], "flag": np.nonzero(cg["flag"] == gi)[0]} for gi in mi]
    cg["wave_val"] = np.nonzero(cg["val"] == gcfg["group_order"].index("wave"))[0]
    cg["wave_flag"] = np.nonzero(cg["flag"] == gcfg["group_order"].index("wave"))[0]
    cg["tblock_val"] = np.array([vcfg["value_columns"].index(c) for c in icfg["design"]["target_block_values"]])
    cg["tblock_flag"] = np.array([vcfg["flag_columns"].index(c) for c in icfg["design"]["target_block_flags"]])
    assert sorted(cg["tblock_val"].tolist()) == sorted(np.concatenate([m["val"] for m in cg["met"]]).tolist())
    assert sorted(cg["tblock_flag"].tolist()) == sorted(np.concatenate([m["flag"] for m in cg["met"]]).tolist())
    return cg


def env(region: str) -> dict:
    dcfg, vcfg, gcfg, icfg = C.load("data"), C.load("views"), C.load("sensor_groups"), C.load("views_groups")
    if region == C.TWELVE:
        vcfg["max_tokens"] = C.cached("socal12")["max_tokens"]
    obs, stations, times, lat, lon = D.open_obs(C.obs(region))
    times_s = times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    norm = json.loads(C.norm(region).read_text())["constants"]
    tab = station_table(obs, times_s, lat, lon, norm, C.base_region(region), dcfg, vcfg)
    return {"region": region, "stations": stations, "tab": tab, "obs": obs, "dcfg": dcfg, "vcfg": vcfg, "gcfg": gcfg, "icfg": icfg,
            "cm": colmap(gcfg, vcfg, icfg)}


def _neighbour_features(tab: dict, target: np.ndarray, t: np.ndarray, keep: np.ndarray,
                        gmask: np.ndarray | None, cm: dict, vcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    val_m, flag_m = width_masks(WIDTH, vcfg)
    vals = tab["vals"][:, t, :].transpose(1, 0, 2) * val_m
    flags = tab["flags"][:, t, :].transpose(1, 0, 2) & flag_m
    if gmask is not None:
        vals = np.where(gmask[:, :, cm["val"]], np.float32(0.0), vals)
        flags = flags & ~gmask[:, :, cm["flag"]]
    present = keep & flags.any(axis=2)
    present[np.arange(len(t)), target] = False
    N, S = present.shape
    feat = np.zeros((N, S, F), np.float32)
    feat[:, :, C_VAL:C_FLAG] = vals
    feat[:, :, C_FLAG:C_CAL] = flags
    feat[:, :, C_CAL:C_GEO] = tab["cal"][t][:, None, :]
    feat[:, :, C_GEO:C_TGT] = tab["geo"][None, :, :]
    feat[~present] = 0.0
    return feat, present


def query(tab: dict, target: np.ndarray, t: np.ndarray, tmet: np.ndarray | None, with_target: bool, cm: dict) -> np.ndarray:
    q = _query(tab, target, t)
    if not with_target:
        return q
    assert tmet is not None
    tv, tf = tab["vals"][target, t], tab["flags"][target, t]
    for k, m in enumerate(cm["met"]):
        on = tmet[:, k:k + 1]
        q[:, C_VAL + m["val"]] = np.where(on, tv[:, m["val"]], np.float32(0.0))
        q[:, C_FLAG + m["flag"]] = np.where(on, tf[:, m["flag"]], False)
    assert not q[:, C_VAL + cm["wave_val"]].any() and not q[:, C_FLAG + cm["wave_flag"]].any(), \
        "gate: the wave columns of the query token must stay zero"
    return q


def tokens_set(tab, target, t, keep, gmask, tmet, with_target, cm, vcfg) -> tuple[np.ndarray, np.ndarray]:
    feat, present = _neighbour_features(tab, target, t, keep, gmask, cm, vcfg)
    N, S = present.shape
    M = vcfg["max_tokens"]
    assert S <= M, "more stations than token slots"
    order = np.argsort(~present, axis=1, kind="stable")
    tok = np.zeros((N, M, F), np.float32)
    valid = np.zeros((N, M), bool)
    tok[:, 0] = query(tab, target, t, tmet, with_target, cm)
    valid[:, 0] = True
    rows = np.arange(N)[:, None]
    tok[:, 1:S] = feat[rows, order[:, :S - 1]]
    valid[:, 1:S] = present[rows, order[:, :S - 1]]
    tok[~valid] = 0.0
    return tok, valid


def tokens_fixed(tab, target, t, keep, gmask, tmet, with_target, cm, vcfg) -> np.ndarray:
    feat, present = _neighbour_features(tab, target, t, keep, gmask, cm, vcfg)
    N, S = present.shape
    tok = np.zeros((N, vcfg["max_tokens"], F), np.float32)
    tok[:, :S] = feat
    tok[np.arange(N), target] = query(tab, target, t, tmet, with_target, cm)
    return tok


def design(tab, g: int, t, keep, gmask, tmet, with_target, cm, vcfg) -> np.ndarray:
    target = np.full(len(t), g)
    feat, _ = _neighbour_features(tab, target, t, keep, gmask, cm, vcfg)
    val_m, flag_m = width_masks(WIDTH, vcfg)
    cols = np.concatenate([np.nonzero(val_m)[0] + C_VAL, np.nonzero(flag_m)[0] + C_FLAG])
    nb = [s for s in range(tab["S"]) if s != g]
    parts = [feat[:, nb][:, :, cols].reshape(len(t), -1)]
    if with_target:
        q = query(tab, target, t, tmet, with_target, cm)
        parts.append(np.concatenate([q[:, C_VAL + cm["tblock_val"]], q[:, C_FLAG + cm["tblock_flag"]]], axis=1))
    parts.append(tab["cal"][t])
    return np.concatenate(parts, axis=1)


def make_inputs(name: str, tab: dict, vcfg: dict, cm: dict, with_target: bool, s: dict, a: int, b: int,
                g: int | None) -> tuple:
    tg, t, keep = s["target"][a:b], s["t"][a:b], s["keep"][a:b]
    gm = None if s.get("gmask") is None else s["gmask"][a:b]
    tm = None if s.get("tmet") is None else s["tmet"][a:b]
    if with_target and tm is None:
        tm = np.zeros((len(t), len(cm["met"])), bool)
    if name == "attention":
        return tokens_set(tab, tg, t, keep, gm, tm, with_target, cm, vcfg)
    if name == "token_mlp":
        return (tokens_fixed(tab, tg, t, keep, gm, tm, with_target, cm, vcfg),)
    return (design(tab, g, t, keep, gm, tm, with_target, cm, vcfg).astype(np.float32),)
