from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oceanbuoy import config as C, data as D, io, states as ST


def main() -> None:
    dcfg, gcfg = C.load("data"), C.load("sensor_groups")
    obs, stations, _, _, _ = D.open_obs(C.obs(C.TWELVE))
    rs = ST.record_states(obs, gcfg, dcfg["variables"], D.split_slice(dcfg, "train"), stations)
    out = {"stations": stations, "met_groups": gcfg["met_groups"], "pi_b_train": rs["train_fractions"].tolist(), "pi_b": rs["all_fractions"].tolist(),
           "train_pool_counts": rs["train_counts"].tolist(), "meta": io.run_meta({"data": dcfg, "sensor_groups": gcfg}, [C.obs(C.TWELVE)])}
    io.write_json(C.outage_patterns(C.TWELVE), out)
    print("->", C.outage_patterns(C.TWELVE))


if __name__ == "__main__":
    main()
