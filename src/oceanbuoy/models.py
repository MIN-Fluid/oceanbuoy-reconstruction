from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rng as R


MODEL_IDS = {"token_mlp": 1, "mlp": 2, "attention": 3}


def gelu(cfg: dict) -> nn.GELU:
    return nn.GELU(approximate=cfg["gelu_approximate"])


class MLPBody(nn.Module):

    def __init__(self, d_in: int, hidden: int, common: dict):
        super().__init__()
        h, hh, p = hidden, common["head_hidden"], common["dropout"]
        self.net = nn.Sequential(
            nn.Linear(d_in, h), gelu(common), nn.Dropout(p),
            nn.Linear(h, h), gelu(common), nn.Dropout(p),
            nn.Linear(h, hh), gelu(common),
            nn.Linear(hh, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x)).squeeze(-1)


class EncoderBlock(nn.Module):

    def __init__(self, a: dict, common: dict):
        super().__init__()
        d = a["d_model"]
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, a["n_heads"], dropout=a["attn_dropout"], bias=True, batch_first=True)
        self.drop1 = nn.Dropout(common["dropout"])
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, a["d_ff"]), gelu(common), nn.Linear(a["d_ff"], d))
        self.drop2 = nn.Dropout(common["dropout"])

    def forward(self, x: torch.Tensor, pad: torch.Tensor, need_weights: bool = False):
        h = self.ln1(x)
        att, w = self.attn(h, h, h, key_padding_mask=pad, need_weights=need_weights, average_attn_weights=False)
        x = x + self.drop1(att)
        x = x + self.drop2(self.ff(self.ln2(x)))
        return x, w


def configure_torch(mcfg: dict) -> None:
    d = mcfg["determinism"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", d["cublas_workspace_config"])
    torch.use_deterministic_algorithms(d["use_deterministic_algorithms"])
    torch.backends.cudnn.benchmark = d["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = d["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = d["allow_tf32"]
    assert d["sdp_backend"] == "math"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(d["mha_fastpath"])
    torch.set_num_threads(d["cpu_threads"])
    omp = os.environ.get("OMP_NUM_THREADS")
    assert omp is None or int(omp) == d["cpu_threads"], f"OMP_NUM_THREADS={omp} but config says {d['cpu_threads']}"


class TokenMLP(nn.Module):

    def __init__(self, mcfg: dict, n_slots: int = 10, n_cols: int = 22):
        super().__init__()
        self.body = MLPBody(n_slots * n_cols, mcfg["token_mlp"]["hidden"], mcfg["common"])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.body(tokens.flatten(1))


class PerStationMLP(nn.Module):

    def __init__(self, mcfg: dict, d_in: int):
        super().__init__()
        self.body = MLPBody(d_in, mcfg["mlp"]["hidden"], mcfg["common"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Attention(nn.Module):

    def __init__(self, mcfg: dict, n_cols: int = 22):
        super().__init__()
        a, c = mcfg["attention"], mcfg["common"]
        d = a["d_model"]
        self.embed = nn.Linear(n_cols, d)
        self.blocks = nn.ModuleList([EncoderBlock(a, c) for _ in range(a["n_layers"])])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, c["head_hidden"]), gelu(c), nn.Linear(c["head_hidden"], 1))

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, need_weights: bool = False):
        x = self.embed(tokens)
        pad = ~valid
        weights = []
        for blk in self.blocks:
            x, w = blk(x, pad, need_weights)
            weights.append(w)
        y = F.softplus(self.head(self.ln_f(x[:, 0]))).squeeze(-1)
        return (y, weights) if need_weights else y


def build_model(name: str, mcfg: dict, seed: int, d_in: int | None = None, target: int = 0) -> nn.Module:
    init_seed = int(R.derive(seed, "init", MODEL_IDS[name], target).integers(2**62))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        if name == "token_mlp":
            model = TokenMLP(mcfg)
        elif name == "mlp":
            assert d_in is not None, "mlp needs d_in"
            model = PerStationMLP(mcfg, d_in)
        elif name == "attention":
            model = Attention(mcfg)
        else:
            raise ValueError(f"unknown model {name!r}")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
