from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, data as D, io


def main() -> None:
    dcfg = C.load("data")
    splits = {s: D.split_slice(dcfg, s) for s in ("train", "val", "test")}
    out = {"rule": {"low_m": D.LOW_M, "min_run_h": D.MIN_RUN_H, "nb_min_m": D.NB_MIN_M}, "regions": {}}
    for region in C.REGIONS:
        obs, st, times, _, _ = D.open_obs(C.obs_unscreened(region))
        v = dcfg["variables"]
        hs, dpd = obs[:, :, v.index("WVHT")].astype(np.float64), obs[:, :, v.index("DPD")].astype(np.float64)
        runs = []
        for g in range(len(st)):
            for s, n, nb in D.wave_fault_runs(hs, dpd, g):
                split = [k for k, sl in splits.items() if sl.start <= s < sl.stop]
                runs.append({"station": st[g], "from": str(times[s])[:13], "to": str(times[s + n - 1])[:13], "hours": n,
                             "neighbour_median_m": nb, "split": split[0] if split else "outside"})
        out["regions"][region] = {"runs": runs, "hours": int(sum(r["hours"] for r in runs))}
        print(f"{region}: {len(runs)} runs, {out['regions'][region]['hours']} h")
    io.write_json(C.suspect_wave_runs(), out)


if __name__ == "__main__":
    main()
