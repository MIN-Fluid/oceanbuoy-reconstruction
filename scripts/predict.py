from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, predict as PR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ridge", "mlp", "token_mlp", "attention"])
    ap.add_argument("--region", required=True, choices=C.REGIONS)
    a = ap.parse_args()
    print("->", PR.predict_test(a.method, a.region, C.seeds(), C.Layout()))


if __name__ == "__main__":
    main()
