"""National Blend of Models (NBM, NOAA, CONUS 2.5 km) snowfall at each station.

NBM is NOAA's calibrated blend of a few dozen models, the forecast WPC and
the NWS offices start from, with its own snowfall field (ASNOW).
PowderBench runs it live as baseline-nbm through Open-Meteo, but deep-powder
had no NBM history to learn from. The archive since 2020-10 is free on AWS
Open Data (s3://noaa-nbm-grib2-pds, no key), and each GRIB2 file has an
.idx index, so one field is a ~1 MB range read rather than the ~185 MB file.

Run rule: like ens_features.init_for_round, the freshest 00/06/12/18 UTC
cycle that is published (LATENCY_H) at least DATA_MARGIN_H before the
league's cutoff, so training and live see the same lead time.

Field: the deterministic 6-hour ASNOW buckets f006..f096, each spread
evenly over its six hours and summed over the round's local-day windows.
CONUS stations of stations/observers only.

Columns (inches of snow), per station, round, horizon:
  nbmx_ctr, nbmx_max, nbmx_mean9   NBM snowfall in the window: own cell,
                                   snowiest and mean of the 3x3 cells around it
  nbmx_last12                      own cell, last 12 local hours of the window
  nbmx_pre12, nbmx_post6           own cell, just before / just after it
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from . import HORIZONS
from .nbhd import _local_hours

NAME = "nbm"
PREFIX = "nbmx"
TAG = "nbhd"  # same release as the other harvested inputs
FILES = "nbmx"
LEAGUES = ("stations", "observers")
BUCKET = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
ARCHIVE_START = pd.Timestamp("2020-10-01")
INIT_HOURS = (0, 6, 12, 18)
LATENCY_H = 2.0  # core files post ~1.3 h after the cycle
STEPS = tuple(range(6, 97, 6))  # 18z run, Pacific stations: the 72 h window ends at f086
IN_PER_M = 39.3701
OFFSETS = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]  # index 4 is the station's own cell
UA = {"User-Agent": "deep-powder (github.com/andrewnakas/deep-powder)"}
RETRIES = 4
_GRID: dict[tuple, tuple[np.ndarray, int, int]] = {}


def init_for_round(round_date: date, league: str) -> pd.Timestamp:
    """ens_features.init_for_round with NBM's cycles and latency."""
    from . import use_pb_data
    from .ens_features import DATA_MARGIN_H

    use_pb_data()
    from powderbench.leagues import get_league

    cutoff = pd.Timestamp(get_league(league).cutoff_utc(round_date)).tz_localize(None)
    latest = cutoff - pd.Timedelta(hours=DATA_MARGIN_H + LATENCY_H)
    day = latest.normalize()
    for back in range(0, 3):
        for h in sorted(INIT_HOURS, reverse=True):
            t = day - pd.Timedelta(days=back) + pd.Timedelta(hours=h)
            if t <= latest:
                return t
    return day - pd.Timedelta(days=3)


def _url(init: pd.Timestamp, step: int) -> str:
    return f"{BUCKET}/blend.{init:%Y%m%d}/{init:%H}/core/blend.t{init:%H}z.core.f{step:03d}.co.grib2"


def _get(url: str, headers: dict | None = None) -> requests.Response | None:
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers={**UA, **(headers or {})}, timeout=120)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == RETRIES - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return None


def byte_range(idx: str, step: int) -> tuple[int, int | None] | None:
    """Offsets of the deterministic 6 h ASNOW message in one file's .idx."""
    want = f":ASNOW:surface:{step - 6}-{step} hour acc fcst:"
    lines = [ln for ln in idx.splitlines() if ln.strip()]
    for k, ln in enumerate(lines):
        if ln.endswith(want):
            start = int(ln.split(":")[1])
            end = int(lines[k + 1].split(":")[1]) - 1 if k + 1 < len(lines) else None
            return start, end
    return None


def _field(init: pd.Timestamp, step: int) -> bytes | None:
    url = _url(init, step)
    idx = _get(url + ".idx")
    if idx is None:
        return None
    rng = byte_range(idx.text, step)
    if rng is None:
        return None
    start, end = rng
    r = _get(url, {"Range": f"bytes={start}-{'' if end is None else end}"})
    return None if r is None else r.content


def decode(msg: bytes) -> tuple[np.ndarray, tuple]:
    """Values (metres of snow, NaN where missing) and a key for the grid."""
    import eccodes

    gid = eccodes.codes_new_from_message(msg)
    try:
        eccodes.codes_set(gid, "missingValue", 1e20)
        vals = np.asarray(eccodes.codes_get_values(gid), "float64")
        if eccodes.codes_get(gid, "alternativeRowScanning"):
            # NBM stores every other row reversed (boustrophedon); eccodes hands the values
            # back in that stored order but the latitudes/longitudes in map order
            ni_ = eccodes.codes_get(gid, "Ni")
            rows = vals.reshape(-1, ni_)
            rows[1::2] = rows[1::2, ::-1].copy()
            vals = rows.reshape(-1)
        key = (eccodes.codes_get(gid, "gridType"), eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj"),
               round(eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"), 4),
               round(eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"), 4))
        if key not in _GRID:
            lat = eccodes.codes_get_array(gid, "latitudes")
            lon = eccodes.codes_get_array(gid, "longitudes")
            _GRID[key] = (_unit(lat, lon), int(key[1]), int(key[2]))
    finally:
        eccodes.codes_release(gid)
    vals[vals > 1e19] = np.nan
    return vals, key


def _unit(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, "float64")), np.radians(np.asarray(lon, "float64"))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def cells(key: tuple, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Flat indices (station, 9) of each station's cell and its 3x3 neighbours."""
    from scipy.spatial import cKDTree

    xyz, ni, nj = _GRID[key]
    if key + ("tree",) not in _GRID:
        _GRID[key + ("tree",)] = cKDTree(xyz)
    tree = _GRID[key + ("tree",)]
    _, flat = tree.query(_unit(lat, lon))
    j, i = np.divmod(flat, ni)
    return np.stack([np.clip(j + a, 0, nj - 1) * ni + np.clip(i + b, 0, ni - 1) for a, b in OFFSETS], 1)


def _read(init: pd.Timestamp, st: pd.DataFrame):
    """Hourly snow (station, 9, hour) in inches and each hour's midpoint."""
    hours, mids, idx, key0 = [], [], None, None
    for step in STEPS:
        msg = _field(init, step)
        if msg is None:
            if step <= 72:
                return None  # the run is incomplete for the windows
            break
        vals, key = decode(msg)
        if idx is None or key != key0:
            idx, key0 = cells(key, st["latitude"].to_numpy(), st["longitude"].to_numpy()), key
        per_h = np.nan_to_num(vals[idx]).clip(min=0) * IN_PER_M / 6.0  # (n, 9)
        for k in range(6):
            hours.append(per_h)
            mids.append(init + pd.Timedelta(hours=step - 6 + k + 0.5))
    return np.stack(hours, 2).astype("float32"), np.array(mids, dtype="datetime64[ns]")


def features(init: pd.Timestamp, st: pd.DataFrame, round_date: date) -> pd.DataFrame:
    got = _read(init, st)
    if got is None:
        return pd.DataFrame()
    snow, mid = got
    r = _local_hours(mid, st, round_date)  # (n, L)
    c = snow[:, 4, :]
    rows = []
    for h in HORIZONS:
        w = (r >= 0) & (r < h)
        full = w.sum(1) >= h - 1
        tot = (snow * w[:, None, :]).sum(2)  # (n, 9)
        f = pd.DataFrame({
            "ctr": tot[:, 4], "max": tot.max(1), "mean9": tot.mean(1),
            "last12": (c * (w & (r >= h - 12))).sum(1),
            "pre12": (c * ((r >= -12) & (r < 0))).sum(1), "post6": (c * ((r >= h) & (r < h + 6))).sum(1),
        }, index=st.index).astype("float32")
        f[~full] = np.nan
        f.columns = [f"{PREFIX}_{k}" for k in f.columns]
        f["horizon_h"] = h
        rows.append(f.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["round_date"] = round_date
    return out.dropna(subset=[f"{PREFIX}_ctr"])


def for_round(league: str, round_date: date, st: pd.DataFrame | None = None, latest: bool = False) -> pd.DataFrame:
    from . import contract
    from .sources.dynamical import in_conus

    if league not in LEAGUES:
        return pd.DataFrame()
    st = contract.stations(league) if st is None else st
    st = st[in_conus(st["latitude"], st["longitude"])]
    if st.empty:
        return pd.DataFrame()
    init = init_for_round(round_date, league)
    if init < ARCHIVE_START:
        return pd.DataFrame()
    tries = [init - pd.Timedelta(hours=6 * k) for k in range(3)] if latest else [init]
    for t in tries:  # live: an older cycle stands in for a late one (same windows)
        out = features(t, st, round_date)
        if len(out):
            return out
    return pd.DataFrame()


def asset(league: str, season: int) -> str:
    return f"{FILES}__{league}__{int(season)}.parquet"


def join(df: pd.DataFrame) -> pd.DataFrame:
    from .nbhd import join_files

    return join_files(df, FILES, PREFIX, "NBM snowfall")
