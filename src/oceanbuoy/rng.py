from __future__ import annotations

import numpy as np

PURPOSES = {"data": 1, "mask": 2, "init": 3, "dropout": 4, "valmask": 5, "boot": 6, "tcopy": 202}


def derive(seed: int, purpose: str, *keys: int) -> np.random.Generator:
    assert all(isinstance(k, (int, np.integer)) and k >= 0 for k in keys), "keys must be non-negative ints"
    return np.random.default_rng(np.random.SeedSequence([int(seed), PURPOSES[purpose], *map(int, keys)]))
