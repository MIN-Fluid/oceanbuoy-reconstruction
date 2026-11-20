from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C, io, models as M, rng as R, samples as SA, validation as VA, views as V


def inputs_fn(name: str, env: dict, g: int | None = None):
    return lambda s, a, b: V.make_inputs(name, env["tab"], env["vcfg"], env["cm"], True, s, a, b, g)


def lr_factor(step: int, warmup: int, max_steps: int, end_factor: float) -> float:
    if step < warmup:
        return step / warmup
    p = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return end_factor + (1.0 - end_factor) * 0.5 * (1.0 + math.cos(math.pi * p))


def schedule_for(name: str, n_samples: int, tcfg: dict) -> dict:
    c = tcfg["neural"]
    steps = n_samples // c["batch_size"]
    if name != "mlp":
        assert steps == c["max_steps"], f"sample list gives {steps} steps, config says {c['max_steps']}"
        return {"steps": steps, "warmup": c["warmup_steps"], "eval_interval": c["eval_interval"],
                "min_steps": c["min_steps"], "patience": c["patience_evals"]}
    f = steps / c["max_steps"]
    return {"steps": steps, "warmup": max(1, round(c["warmup_steps"] * f)),
            "eval_interval": max(1, round(c["eval_interval"] * f)),
            "min_steps": max(1, round(c["min_steps"] * f)), "patience": c["patience_evals"]}


def to_device(arrays, device) -> tuple:
    return tuple(torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in arrays)


def cache_inputs(inputs, s: dict, device, batch: int) -> list:
    n = len(s["t"])
    return [to_device(inputs(s, a, min(a + batch, n)), device) for a in range(0, n, batch)]


def predict_cached(model, cache: list) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for x in cache:
            out.append(model(*x).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


PARTS = {"embed": "embed.", "blocks": "blocks.", "ln_f": "ln_f.", "head": "head."}


def _trainable(model, parts):
    if parts is None:
        return list(model.parameters())
    prefixes = tuple(PARTS[p] for p in parts)
    out = []
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(prefixes))
        if p.requires_grad:
            out.append(p)
    assert out, f"no parameter matches {parts}"
    return out


def fit(name: str, env: dict, train_s: dict, val, seed: int, sch: dict, device, g: int | None = None, init_state: dict | None = None,
        trainable: list[str] | None = None, dropout_key: int | None = None, lr: float | None = None, max_steps: int | None = None,
        d_in: int | None = None, log=print) -> dict:
    tcfg, mcfg = C.load("train"), C.load("models")
    c = tcfg["neural"]
    B = c["batch_size"]
    inputs = inputs_fn(name, env, g)
    y_train = V.truth(env["obs"], train_s["target"], train_s["t"], env["dcfg"])
    model = M.build_model(name, mcfg, seed, d_in=d_in, target=g or 0).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    params = _trainable(model, trainable)
    opt = torch.optim.AdamW(params, lr=c["lr"] if lr is None else lr, betas=tuple(c["betas"]), eps=c["eps"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, sch["warmup"], sch["steps"], c["end_factor"]))
    key = (g or 0) if dropout_key is None else dropout_key
    torch.manual_seed(int(R.derive(seed, "dropout", M.MODEL_IDS[name], key).integers(2**62)))
    t_cache = time.time()
    vcache = cache_inputs(inputs, val.s, device, c["val_batch"])
    cache_min = round((time.time() - t_cache) / 60, 2)
    steps = sch["steps"] if max_steps is None else min(max_steps, sch["steps"])
    best, best_step, since, history, t0, best_state = math.inf, 0, 0, [], time.time(), None
    CH = c["chunk_batches"] * B
    chunk_a, chunk = -1, None
    for s in range(1, steps + 1):
        a = (s - 1) * B
        if a // CH != chunk_a:
            chunk_a = a // CH
            ca, cb = chunk_a * CH, min((chunk_a + 1) * CH, steps * B)
            chunk = (ca, to_device(inputs(train_s, ca, cb), device), torch.from_numpy(y_train[ca:cb]).to(device))
        ca, xs, ys = chunk
        sl = slice(a - ca, a - ca + B)
        model.train()
        loss = F.mse_loss(model(*(x[sl] for x in xs)), ys[sl])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % sch["eval_interval"] == 0 or s == steps:
            rmse = val.score(predict_cached(model, vcache))
            history.append({"step": s, "loss": loss.item(), "val_rmse_m": rmse, "lr": opt.param_groups[0]["lr"],
                            "minutes": round((time.time() - t0) / 60, 2)})
            if rmse < best:
                best, best_step, since = rmse, s, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
            log(f"  step {s:6d}  loss {loss.item():.4f}  val {rmse:.4f} m  best {best:.4f} @ {best_step}")
            if s >= sch["min_steps"] and since >= sch["patience"]:
                log(f"  early stop at {s}; best {best:.4f} m at step {best_step}")
                break
    return {"state": best_state, "best_step": best_step, "best": best, "steps_run": s, "history": history,
            "minutes": round((time.time() - t0) / 60, 2), "val_cache_minutes": cache_min,
            "params": M.count_params(model) if trainable is None else sum(p.numel() for p in model.parameters()),
            "n_trainable": sum(p.numel() for p in params)}


def _device():
    return torch.device(C.load("train")["neural"]["device"] if torch.cuda.is_available() else "cpu")


def _save(out: Path, state, meta, history) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save({"state_dict": state, "meta": meta, "history": history}, tmp)
    os.replace(tmp, out)


def _meta(name, region, seed, g, r, sch, train_s, val, files, env, extra=None) -> dict:
    return {"model": name, "region": region, "seed": seed, "target": g, "target_station": None if g is None else env["stations"][g],
            "params": r["params"], "n_trainable": r["n_trainable"], "schedule": sch, "steps_run": r["steps_run"], "best_step": r["best_step"],
            "best_val_rmse_m": r["best"], "minutes": r["minutes"], "val_cache_minutes": r["val_cache_minutes"],
            "n_train_samples": int(len(train_s["t"])), "n_val_rows": int(val.n_rows), "n_val_items": int(val.n_items),
            "val_criterion": "sqrt(mean_rows sum_p pi_p e_p^2)", "files_sha256": {k: io.sha256_file(v) for k, v in files.items()},
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
            "cuda": torch.version.cuda, "torch_threads": torch.get_num_threads(), **(extra or {}),
            **io.run_meta({"pipeline": C.load("pipeline"), "train": C.load("train"), "models": C.load("models"), "views": env["vcfg"],
                           "data": env["dcfg"]}, [])}


def load_inputs(region: str, seed: int, L: C.Layout, env: dict):
    sp, vp = L.stream(region, seed), L.val(region)
    train_s = SA.load_samples(sp, env["gcfg"])
    val = VA.load_recval(vp, env["obs"], env["dcfg"])
    return train_s, val, {"stream": sp, "val": vp}


def train_net(name: str, region: str, seed: int, L: C.Layout, target: int | None = None, max_steps: int | None = None, log=print) -> Path:
    assert name in ("attention", "token_mlp", "mlp") and (region != C.TWELVE or name == "attention")
    M.configure_torch(C.load("models"))
    env = V.env(region)
    train_s, val, files = load_inputs(region, seed, L, env)
    d_in, g = None, None
    if name == "mlp":
        g = int(target)
        train_s = SA.subset(train_s, np.nonzero(train_s["target"] == g)[0])
        val = val.subset_target(g)
        d_in = inputs_fn(name, env, g)(train_s, 0, 1)[0].shape[1]
    sch = schedule_for(name, len(train_s["t"]), C.load("train"))
    r = fit(name, env, train_s, val, seed, sch, _device(), g=g, max_steps=max_steps, d_in=d_in, log=log)
    out = L.expansion("retrain", seed) if region == C.TWELVE else L.ckpt(name, region, seed, g)
    _save(out, r["state"], _meta(name, region, seed, g, r, sch, train_s, val, files, env), r["history"])
    return out


def train_finetune(variant: str, seed: int, L: C.Layout, max_steps: int | None = None, log=print) -> Path:
    fcfg = C.load("finetune")
    v = fcfg["variants"][variant]
    M.configure_torch(C.load("models"))
    tcfg = C.load("train")
    env = V.env(C.TWELVE)
    train_s, val, files = load_inputs(C.TWELVE, seed, L, env)
    n = fcfg["max_steps"] * tcfg["neural"]["batch_size"]
    train_s = SA.subset(train_s, np.arange(n))
    sch = schedule_for("mlp", n, tcfg)
    src = C.Layout().ckpt("attention", "socal", seed)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    assert ck["meta"]["seed"] == seed, "the fine-tune starts from the Southern California attention model of the same seed"
    r = fit("attention", env, train_s, val, seed, sch, _device(), init_state=ck["state_dict"], trainable=v["trainable"],
            dropout_key=v["dropout_key"], lr=fcfg["lr"], max_steps=max_steps, log=log)
    out = L.expansion(variant, seed)
    _save(out, r["state"], _meta("attention", C.TWELVE, seed, None, r, sch, train_s, val, files, env,
                                 {"kind": "finetune", "variant": variant, "init_from": str(src.relative_to(C.ROOT)),
                                  "init_sha256": io.sha256_file(src), "trainable": v["trainable"], "lr": fcfg["lr"]}), r["history"])
    return out
