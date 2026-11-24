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


def _pairs(names: list[str]) -> list[tuple[str, str]]:
    return [("attention", m) for m in names if m != "attention"]


def evaluate_regions(cfg: dict) -> dict:
    """Sections 4.2-4.3 and Appendix B of the paper. Complete outages (pattern 0) on every test sample, the interpolators on the samples
    with two or more reporting neighbors; partial outages (patterns 1-7 weighted by the training-year shares) at the NDBC buoys."""
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
            sq[m], msq[m] = EV.complete_sq(P[m]["pred"], y)
        ip = {}
        for m in INTERP:
            d = EV.read_predictions(PRED / m / f"{region}.nc")
            assert np.array_equal(d["target"], tg) and np.array_equal(d["t"], t)
            ip[m] = (d["pred"] - y) ** 2
        ge2 = np.nonzero(nl >= 2)[0]
        boot = EV.Bootstrap(t, y, region, cfg)
        reg = {"stations": stations, "all": EV.score(boot, sq, _pairs(list(LEARNED)), msq)}
        reg["all"]["best_baseline"] = min((m for m in LEARNED if m != "attention"), key=lambda m: reg["all"]["methods"][m]["rmse_m"])
        boot2 = EV.Bootstrap(t[ge2], y[ge2], region, cfg)
        reg["two_or_more_neighbors"] = EV.score(boot2, {m: v[ge2] for m, v in (ip | sq).items()}, [])
        groups = []
        for lo, hi in EV.group_bins(nl, cfg["min_group_n"]):
            r = np.nonzero((nl >= lo) & (nl <= hi))[0]
            g_sq = {m: v[r] for m, v in sq.items()} | ({m: v[r] for m, v in ip.items()} if lo >= 2 else {})
            gb = EV.Bootstrap(t[r], y[r], region, cfg)
            first = EV.score(gb, g_sq, [])
            best = min((m for m in g_sq if m != "attention"), key=lambda m: first["methods"][m]["rmse_m"])
            sc = EV.score(gb, g_sq, [("attention", best)])
            groups.append({"n_neighbors": [lo, hi], "best_baseline": best, **sc})
        reg["groups"] = groups
        psq = {m: EV.partial_sq(P[m]["pred"], y, W) for m in LEARNED}
        ok = psq["attention"][2]
        tb = EV.Bootstrap(t[ok], y[ok], region, cfg)
        by_type = {}
        for cond, src in (("complete", {m: sq[m][ok] for m in LEARNED}), ("partial", {m: psq[m][0][ok] for m in LEARNED})):
            sc = EV.score(tb, src, _pairs(list(LEARNED)))
            sc["best_baseline"] = min((m for m in LEARNED if m != "attention"), key=lambda m: sc["methods"][m]["rmse_m"])
            by_type[cond] = sc
        by_type["partial-complete"] = {m: EV.score(tb, {"complete": sq[m][ok], "partial": psq[m][0][ok]}, [("partial", "complete")])["differences"]["partial-complete"]
                                       for m in LEARNED}
        reg["outage_type"] = by_type
        sigma = float(y.std())
        reg["nrmse"] = {"sigma_m": sigma, "mean_pairwise_r": mean_pairwise_r(region),
                        **{m: reg["all"]["methods"][m]["rmse_m"] / sigma for m in LEARNED}}
        out[region] = reg
        print(region, {m: round(reg["all"]["methods"][m]["rmse_m"], 4) for m in LEARNED}, flush=True)
    return out


def evaluate_added_buoys(cfg: dict) -> dict:
    """Section 4.4 of the paper: the ten original buoys (complete outages on every sample, partial outages at the six NDBC buoys) and the
    two added buoys (complete outages; partial outages at 46047, the CDIP buoy 46239 having none)."""
    pi, _ = pi_train("socal")
    head = EV.read_predictions(PRED / "attention" / "socal.nc")
    y, t, tg = head["truth"], head["t"], head["target"]
    W = EV.pattern_weights(pi, tg)
    preds = {"base": head["pred"]}
    for name in LADDER:
        d = EV.read_predictions(PRED / "expansion" / f"{name}.nc")
        assert np.array_equal(d["target"], tg) and np.array_equal(d["t"], t) and d["seeds"] == head["seeds"]
        preds[name] = d["pred"]
    boot = EV.Bootstrap(t, y, "socal", cfg)
    original = {"complete": EV.score(boot, {m: EV.complete_sq(v, y)[0] for m, v in preds.items()}, [(m, "base") for m in LADDER])}
    part = {m: EV.partial_sq(v, y, W) for m, v in preds.items()}
    ok = part["base"][2]
    original["partial"] = EV.score(EV.Bootstrap(t[ok], y[ok], "socal", cfg), {m: v[0][ok] for m, v in part.items()}, [(m, "base") for m in LADDER])
    out = {"original_buoys": original}
    states = json.loads((ROOT / "data" / "outage_patterns" / "socal12.json").read_text())
    pi12 = np.asarray(states["pi_b_train"], float)
    added = {}
    for b, g in ADDED.items():
        comp, par = {}, {}
        for name in ADDED_METHODS:
            d = EV.read_predictions(PRED / "added_buoys" / f"{name}_{b}.nc")
            yb, n = d["truth"], len(d["truth"])
            Wb = [np.full(n, pi12[g, p]) for p in range(8)]
            if d["pred"].ndim == 3:
                comp[name] = EV.complete_sq(d["pred"], yb)[0]
                ps, _, okb = EV.partial_sq(d["pred"], yb, Wb)
                if okb.all():
                    par[name] = ps
            else:
                comp[name] = (d["pred"] - yb) ** 2
                if pi12[g, 1:].sum() > 0:
                    par[name] = comp[name]
            tb = d["t"]
        bb = EV.Bootstrap(tb, yb, "socal", cfg)
        added[b] = {"complete": EV.score(bb, comp, [(m, "none") for m in ADDED_METHODS if m != "none"])}
        if len(par) == len(ADDED_METHODS):
            added[b]["partial"] = EV.score(bb, par, [(m, "none") for m in ADDED_METHODS if m != "none"])
    out["added_buoys"] = added
    return out


def main() -> None:
    cfg = EV.load_score_config(ROOT)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "evaluation_regions.json").write_text(json.dumps(evaluate_regions(cfg), indent=1))
    (out_dir / "evaluation_added_buoys.json").write_text(json.dumps(evaluate_added_buoys(cfg), indent=1))
    print("->", out_dir / "evaluation_regions.json", out_dir / "evaluation_added_buoys.json")


if __name__ == "__main__":
    main()
