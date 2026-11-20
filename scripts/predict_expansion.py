from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, expansion as EX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buoys", required=True, choices=["original", "added"])
    a = ap.parse_args()
    run = EX.original_buoys if a.buoys == "original" else EX.added_buoys
    for p in run(C.seeds(), C.Layout()):
        print("->", p)


if __name__ == "__main__":
    main()
