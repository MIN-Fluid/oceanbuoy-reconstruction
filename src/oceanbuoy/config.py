from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ("socal", "gom", "pnw")
TWELVE = "socal12"
ALL = REGIONS + (TWELVE,)


def load(name: str) -> dict:
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


@lru_cache(maxsize=None)
def cached(name: str) -> dict:
    return load(name)


def seeds() -> list[int]:
    return cached("pipeline")["train_seeds"]


def seed_tag(seed: int) -> str:
    assert 20260800 <= seed <= 20260899, f"seed {seed} outside the 202608xx family"
    return f"s{seed - 20260800:02d}"


def base_region(region: str) -> str:
    return cached("socal12")["base"] if region == TWELVE else region


def region_index(region: str) -> int:
    return cached("socal12")["region_index"] if region == TWELVE else REGIONS.index(region)


def pi_matrix(region: str, stations: list[str], key: str) -> np.ndarray:
    if region == TWELVE:
        J = json.loads(outage_patterns(TWELVE).read_text())
        assert J["stations"] == stations, "data/outage_patterns/socal12.json stations differ from the record"
        m = np.asarray(J[key], float)
    else:
        d = cached("design")[key][region]
        assert list(d) == list(stations), f"{region}: configs/design.json {key} buoys differ from the record"
        m = np.asarray([d[s] for s in stations], float)
    assert m.shape == (len(stations), 8) and np.allclose(m.sum(axis=1), 1.0, atol=1e-12, rtol=0), f"{key} rows must sum to 1"
    return m


def obs_unscreened(region: str) -> Path:
    return ROOT / "data" / "obs_unscreened" / f"{region}.nc"


def obs(region: str) -> Path:
    return ROOT / "data" / "obs" / f"{region}.nc"


def norm(region: str) -> Path:
    return ROOT / "data" / "norm" / f"{region}.json"


def suspect_wave_runs() -> Path:
    return ROOT / "data" / "qc" / "suspect_wave_runs.json"


def test_set(region: str) -> Path:
    return ROOT / "data" / "samples" / "test" / f"{region}.nc"


def outage_patterns(name: str) -> Path:
    return ROOT / "data" / "outage_patterns" / f"{name}.json"


def interp_params(region: str) -> Path:
    return ROOT / "models" / "interp" / f"{region}.json"


def pred(method: str, region: str) -> Path:
    return ROOT / "predictions" / method / f"{region}.nc"


class Layout:
    def __init__(self, root: Path | None = None):
        self.base = ROOT if root is None else Path(root)
        self.samples = self.base / "data" / "samples"
        self.models = self.base / "models"

    def val(self, region: str) -> Path:
        return self.samples / "val" / f"{region}.nc"

    def stream(self, region: str, seed: int) -> Path:
        return self.samples / "train" / f"{region}_{seed_tag(seed)}.nc"

    def ckpt(self, method: str, region: str, seed: int, target: int | None = None) -> Path:
        assert (target is None) == (method != "mlp")
        name = seed_tag(seed) + ("" if target is None else f"_t{target:02d}")
        return self.models / method / region / f"{name}.pt"

    def ridge(self, region: str, seed: int) -> Path:
        return self.models / "ridge" / region / f"{seed_tag(seed)}.nc"

    def expansion(self, variant: str, seed: int) -> Path:
        return self.models / "expansion" / variant / f"{seed_tag(seed)}.pt"
