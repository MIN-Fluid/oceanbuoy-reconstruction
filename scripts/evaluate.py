from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from itertools import combinations

from oceanbuoy import config as C, data as D, evaluation as EV

PRED = ROOT / "predictions"
LEARNED = ("ridge", "mlp", "token_mlp", "attention")
INTERP = ("idw", "ok", "ak")
LADDER = ("none", "ft_emb", "ft_enc", "ft", "retrain")
ADDED = {"46239": 10, "46047": 11}
ADDED_METHODS = ("none", "idw", "ok", "ak", "ft_emb", "ft_enc", "ft", "retrain")


def pi_train(region: str) -> tuple[np.ndarray, list[str]]:
    d = json.loads((ROOT / "configs" / "design.json").read_text())["pi_b_train"][region]
    return np.asarray(list(d.values()), float), list(d)


def mean_pairwise_r(region: str) -> float:
    dcfg = C.load("data")
    lo, hi = dcfg["splits"]["train"]["idx_start"], dcfg["splits"]["test"]["idx_end"]
    obs = D.open_obs(C.obs(region))[0]
    hs = obs[:, :, dcfg["variables"].index("WVHT")].astype(np.float64)[:, lo:hi + 1]
    fin = np.isfinite(hs)
    rr = [float(np.corrcoef(hs[i, fin[i] & fin[j]], hs[j, fin[i] & fin[j]])[0, 1]) for i, j in combinations(range(hs.shape[0]), 2)]
    return float(np.mean(rr))


def section_4_1(cfg: dict) -> dict:
    out = {}
    for region in EV.REGIONS:
        pi, stations = pi_train(region)
        P = {m: EV.read_predictions(PRED / m / f"{region}.nc") for m in LEARNED}
        ref = P["attention"]
        for m in LEARNED:
            for k in ("target", "t", "truth", "n_live"):
                assert np.array_equal(P[m][k], ref[k]), f"{region} {m}: samples differ from the attention model's"
        y, t, tg, nl = ref["truth"], ref["t"], ref["target"], ref["n_live"]
        W = EV.pattern_weights(pi, tg)
        sq, msq = {}, {}
        for m in LEARNED:
            sq[m], msq[m] = EV.weighted_sq(P[m]["pred"], y, W)
        ip = {}
        for m in INTERP:
            d = EV.read_predictions(PRED / m / f"{region}.nc")
            assert np.array_equal(d["target"], tg) and np.array_equal(d["t"], t)
            ip[m] = (d["pred"] - y) ** 2
        ge2 = np.nonzero(nl >= 2)[0]
        boot = EV.Bootstrap(t, y, region, cfg)
        reg = {"stations": stations, "all": EV.score(boot, sq, [("attention", "mlp"), ("attention", "token_mlp")], msq)}
        boot2 = EV.Bootstrap(t[ge2], y[ge2], region, cfg)
        reg["two_or_more_neighbors"] = EV.score(boot2, {m: v[ge2] for m, v in (ip | sq).items()}, [])
        groups = []
        for lo, hi in EV.group_bins(nl, cfg["min_group_n"]):
            r = np.nonzero((nl >= lo) & (nl <= hi))[0]
            g_sq = {m: v[r] for m, v in sq.items()} | ({m: v[r] for m, v in ip.items()} if lo >= 2 else {})
            gb = EV.Bootstrap(t[r], y[r], region, cfg)
            baselines = [m for m in g_sq if m != "attention"]
            first = EV.score(gb, g_sq, [])
            best = min(baselines, key=lambda m: first["methods"][m]["rmse_m"])
            pairs = [("attention", "mlp"), ("attention", "token_mlp")] + ([("attention", best)] if best not in ("mlp", "token_mlp") else [])
            sc = EV.score(gb, g_sq, pairs)
            groups.append({"n_neighbors": [lo, hi], "best_baseline": best, **sc})
        reg["groups"] = groups
        by_type = {}
        for m in LEARNED:
            c, w, ok = EV.outage_type_sq(P[m]["pred"], y, W)
            tb = EV.Bootstrap(t[ok], y[ok], region, cfg)
            sc = EV.score(tb, {"complete": c[ok], "partial": w[ok]}, [("partial", "complete")])
            by_type[m] = {"n_samples": sc["n_samples"], "complete_rmse_m": sc["methods"]["complete"]["rmse_m"],
                          "partial_rmse_m": sc["methods"]["partial"]["rmse_m"], "partial-complete": sc["differences"]["partial-complete"]}
        reg["outage_type"] = by_type
        sigma = float(y.std())
        reg["nrmse"] = {"sigma_m": sigma, "mean_pairwise_r": mean_pairwise_r(region),
                        **{m: reg["all"]["methods"][m]["rmse_m"] / sigma for m in LEARNED}}
        out[region] = reg
        print(region, {m: round(reg["all"]["methods"][m]["rmse_m"], 4) for m in LEARNED}, flush=True)
    return out


def section_4_2(cfg: dict) -> dict:
    pi, _ = pi_train("socal")
    head = EV.read_predictions(PRED / "attention" / "socal.nc")
    y, t, tg = head["truth"], head["t"], head["target"]
    W = EV.pattern_weights(pi, tg)
    sq = {"base": EV.weighted_sq(head["pred"], y, W)[0]}
    for name in LADDER:
        d = EV.read_predictions(PRED / "expansion" / f"{name}.nc")
        assert np.array_equal(d["target"], tg) and np.array_equal(d["t"], t) and d["seeds"] == head["seeds"]
        sq[name] = EV.weighted_sq(d["pred"], y, W)[0]
    boot = EV.Bootstrap(t, y, "socal", cfg)
    out = {"original_buoys": EV.score(boot, sq, [(m, "base") for m in LADDER])}
    states = json.loads((ROOT / "data" / "outage_patterns" / "socal12.json").read_text())
    pi12 = np.asarray(states["pi_b_train"], float)
    added = {}
    for b, g in ADDED.items():
        sqb = {}
        for name in ADDED_METHODS:
            d = EV.read_predictions(PRED / "added_buoys" / f"{name}_{b}.nc")
            yb, n = d["truth"], len(d["truth"])
            Wb = [np.full(n, pi12[g, p]) for p in range(8)]
            if d["pred"].ndim == 3:
                sqb[name] = EV.weighted_sq(d["pred"], yb, Wb)[0]
            else:
                sqb[name] = sum(np.asarray(Wb[p], float) * (d["pred"] - yb) ** 2 for p in range(8))
            tb = d["t"]
        bb = EV.Bootstrap(tb, yb, "socal", cfg)
        added[b] = EV.score(bb, sqb, [(m, "none") for m in ADDED_METHODS if m != "none"])
    out["added_buoys"] = added
    return out


def main() -> None:
    cfg = EV.load_score_config(ROOT)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "evaluation_4_1.json").write_text(json.dumps(section_4_1(cfg), indent=1))
    (out_dir / "evaluation_4_2.json").write_text(json.dumps(section_4_2(cfg), indent=1))
    print("->", out_dir / "evaluation_4_1.json", out_dir / "evaluation_4_2.json")


if __name__ == "__main__":
    main()
