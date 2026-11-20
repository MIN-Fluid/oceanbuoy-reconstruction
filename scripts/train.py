from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, ridge as RR, train as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ridge", "mlp", "token_mlp", "attention", "finetune"])
    ap.add_argument("--region", choices=C.ALL)
    ap.add_argument("--seed", type=int, required=True, choices=C.seeds())
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--variant", choices=sorted(C.load("finetune")["variants"]))
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--root", type=Path, default=None)
    a = ap.parse_args()
    assert a.max_steps is None or a.root is not None, "--max-steps goes with --root"
    L = C.Layout(a.root)
    if a.model == "ridge":
        out = RR.train_ridge(a.region, a.seed, L)
    elif a.model == "finetune":
        out = T.train_finetune(a.variant, a.seed, L, a.max_steps)
    else:
        out = T.train_net(a.model, a.region, a.seed, L, a.target, a.max_steps)
    print("->", out)


if __name__ == "__main__":
    main()
