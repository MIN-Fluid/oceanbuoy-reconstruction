from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import build_samples as BS, config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=C.ALL)
    ap.add_argument("--what", required=True, choices=["val", "train"])
    ap.add_argument("--seed", type=int, choices=C.seeds())
    ap.add_argument("--root", type=Path, default=None)
    a = ap.parse_args()
    L = C.Layout(a.root)
    out = BS.build_val(a.region, L) if a.what == "val" else BS.build_stream(a.region, a.seed, L)
    print(f"{a.region} {a.what} -> {out}")


if __name__ == "__main__":
    main()
