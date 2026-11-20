from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

VARIABLES = ["WVHT", "DPD", "APD", "WDIR", "WSPD", "PRES", "WTMP"]

SENTINELS = {
    "WDIR": 999.0,
    "WSPD": 99.0,
    "GST": 99.0,
    "WVHT": 99.0,
    "DPD": 99.0,
    "APD": 99.0,
    "MWD": 999.0,
    "PRES": 9999.0,
    "ATMP": 999.0,
    "WTMP": 999.0,
    "DEWP": 999.0,
    "VIS": 99.0,
    "TIDE": 99.0,
}

GRID_FREQ = "1h"


TRAIN_END = "2022-12-31 23:59:59"
VAL_END = "2023-12-31 23:59:59"


def _read_one_year(path: Path) -> pd.DataFrame | None:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    lines = raw.decode("ascii", errors="replace").splitlines()
    if len(lines) < 3:
        return None
    header = lines[0].lstrip("#").split()
    body = "\n".join(lines[2:])
    if not body.strip():
        return None
    df = pd.read_csv(io.StringIO(body), sep=r"\s+", names=header, engine="python")
    if df.empty:
        return None

    year_col = "YY" if "YY" in df.columns else "YYYY"
    if "mm" not in df.columns:
        df["mm"] = 0
    try:
        dt = pd.to_datetime(
            dict(year=df[year_col].astype(int), month=df["MM"].astype(int),
                 day=df["DD"].astype(int), hour=df["hh"].astype(int), minute=df["mm"].astype(int)),
            utc=True,
        )
    except Exception:
        return None
    df.index = dt
    df.index.name = "time"

    keep_cols = [c for c in VARIABLES if c in df.columns]
    df = df[keep_cols].apply(pd.to_numeric, errors="coerce")

    for col in keep_cols:
        sentinel = SENTINELS.get(col)
        if sentinel is not None:
            df.loc[np.isclose(df[col].values, sentinel, atol=1e-6), col] = np.nan

    for col in VARIABLES:
        if col not in df.columns:
            df[col] = np.nan
    return df[VARIABLES]


def load_station_raw(stn: str) -> pd.DataFrame:
    stn_dir = RAW_DIR / stn
    files = sorted(stn_dir.glob(f"{stn}h*.txt.gz"))
    frames = []
    for fp in files:
        df = _read_one_year(fp)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No parsable data for station {stn}")
    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="first")]
    full = full.sort_index()
    return full


def _circular_mean_deg(series: pd.Series) -> float:
    obs = series.dropna().values
    if len(obs) == 0:
        return np.nan
    rad = np.deg2rad(obs)
    s, c = np.sin(rad).mean(), np.cos(rad).mean()
    if s == 0 and c == 0:
        return np.nan
    return float(np.rad2deg(np.arctan2(s, c)) % 360.0)


def align_to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=grid, columns=VARIABLES, dtype=float)
    resampler = df.resample(GRID_FREQ, label="left", closed="left")
    for col in VARIABLES:
        if col == "WDIR":
            agg = resampler[col].apply(_circular_mean_deg)
        else:
            agg = resampler[col].mean()
        out[col] = agg.reindex(grid)
    return out


