from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, data as D, io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=C.ALL)
    region = ap.parse_args().region
    cfg = C.load("data")
    if region == C.TWELVE:
        constants = json.loads(C.norm(C.base_region(region)).read_text())["constants"]
    else:
        obs = D.open_obs(C.obs(region))[0]
        constants = D.regional_norm(obs, cfg)
    io.write_json(C.norm(region), {"meta": io.run_meta(cfg, [C.obs(region)]), "constants": constants})
    print(f"{region}: " + ", ".join(f"{v} {c['mean']:.4f}±{c['std']:.4f}" for v, c in constants.items()))


if __name__ == "__main__":
    main()
