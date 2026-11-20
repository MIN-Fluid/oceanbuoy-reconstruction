from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import config as C, data as D, io, samples as SA, sensor_groups as GR, states as ST, validation as VA


def _load(region: str):
    obs, stations, _, _, _ = D.open_obs(C.obs(region))
    dcfg, scfg, gcfg = C.load("data"), C.load("sampling"), C.load("sensor_groups")
    if region == C.TWELVE:
        scfg["stream"]["max_neighbors"] = C.cached("socal12")["max_neighbors"]
    rs = ST.record_states(obs, gcfg, dcfg["variables"], D.split_slice(dcfg, "train"), stations)
    meta = json.dumps(io.run_meta({"pipeline": C.cached("pipeline"), "design": C.cached("design"), "data": dcfg, "sampling": scfg,
                                   "sensor_groups": gcfg}, [C.obs(region)]))
    return obs, stations, dcfg, scfg, gcfg, rs, meta


def build_val(region: str, L: C.Layout) -> Path:
    obs, stations, dcfg, scfg, gcfg, rs, meta = _load(region)
    va = D.split_slice(dcfg, "val")
    observed = np.isfinite(obs[:, :, dcfg["variables"].index("WVHT")])
    v = SA.make_val_set(observed, scfg, va.start, va.stop)
    tg, t = v["target"].astype(np.int64), v["t"].astype(np.int64)
    assert observed[tg, t].all(), "a validation label is NaN"
    pib, pit = C.pi_matrix(region, stations, "pi_b"), C.pi_matrix(region, stations, "pi_b_train")
    assert np.allclose(pib, rs["all_fractions"], atol=1e-15, rtol=0) and np.allclose(pit, rs["train_fractions"], atol=1e-15, rtol=0), \
        "the pattern weights differ from the fractions of the record"
    actual = GR.pattern_code(rs["alive"][tg, t][:, GR.met_index(gcfg)])
    ir, ic, iw = VA.build_items_pib(tg, actual, pit)
    ds = xr.Dataset({"target": ("sample", v["target"]), "t": ("sample", v["t"]), "actual_bits": ("sample", actual),
                     "item_row": ("item", ir.astype(np.int32)), "item_code": ("item", ic), "item_w": ("item", iw)},
                    coords={"station": stations},
                    attrs={"region": region, "split": "val", "pi": json.dumps(pit.tolist()), "meta": meta})
    out = L.val(region)
    io.write_netcdf(ds, out)
    return out


def build_stream(region: str, seed: int, L: C.Layout) -> Path:
    obs, stations, dcfg, scfg, gcfg, rs, meta = _load(region)
    tr = D.split_slice(dcfg, "train")
    observed = np.isfinite(obs[:, :, dcfg["variables"].index("WVHT")])
    st = SA.make_stream(observed, scfg, seed, tr.start, tr.stop)
    tg, t = st["target"].astype(np.int64), st["t"].astype(np.int64)
    assert observed[tg, t].all(), "a sample's label is NaN"
    bits, keep = SA.target_state_bits(tg, t, rs["alive"], rs["train_counts"], gcfg, seed, C.region_index(region))
    zero = np.zeros(len(t), np.uint16)
    ds = xr.Dataset({"target": ("sample", st["target"]), "t": ("sample", st["t"]), "keep_bits": ("sample", SA.pack_bits(keep)),
                     **{k: ("sample", v) for k, v in bits.items()}, **{f"goff_{g}": ("sample", zero) for g in gcfg["group_order"]}},
                    coords={"station": stations}, attrs={"region": region, "seed": seed, "split": "train", "meta": meta})
    out = L.stream(region, seed)
    io.write_netcdf(ds, out)
    return out
