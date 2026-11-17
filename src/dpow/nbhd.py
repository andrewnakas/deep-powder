"""Hourly GFS around each station: grid-neighbourhood and timing features.

The daily ensemble features (ens_features.py) see one grid cell and whole
local days. Two things they cannot say:

- Terrain the 0.25 degree grid smooths over. The 3x3 block of cells around a
  station brackets what its mountain can get: the wettest and coldest
  neighbour, and how much the cells disagree.
- When the snow falls inside the window. An observer reads a board at a
  fixed hour, so snow just before the window opens or just after it closes
  often lands in the neighbouring day's report.

One run of the GFS archive on dynamical.org (hourly steps, no quota) gives
both. The run is the one a forecast made at the league's cutoff can use
(ens_features.init_for_round), in training and live alike.

Columns (inches of snow unless noted), per station, round and horizon:
  gfsx_ctr, gfsx_max, gfsx_mean, gfsx_min, gfsx_sd   window total: own cell and over the 9 cells
  gfsx_liq_max                                       wettest cell, liquid
  gfsx_frz_max, gfsx_tmin                            snowiest cell's frozen share, coldest cell's mean 2 m T
  gfsx_first12, gfsx_last12                          first and last 12 local hours of the window
  gfsx_pre12, gfsx_post6                             just outside the window
  gfsx_peak6, gfsx_hours                             heaviest 6 h, hours with snow
  gfsx_pre6, gfsx_pre24, gfsx_post12, gfsx_first6, gfsx_last6, gfsx_last24   finer and wider edges
  gfsx_liq_pre12, gfsx_rain, gfsx_t_first12, gfsx_t_last12                   liquid before, rain inside, T at each end
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import HORIZONS
from .ens_features import MM_PER_IN, init_for_round, slr_2m, station_tz

NAME = "gfs"
PREFIX = "gfsx"
N_LEAD = 105  # hourly steps read (one chunk of the archive); covers the 72 h window from any usable run
TAG = "nbhd"
OFFSETS = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]  # index 4 is the station's own cell


def _read(init: pd.Timestamp, st: pd.DataFrame, retries: int = 5):
    """-> snow (in), liquid (in), 2 m T, each (station, 9 cells, hour), and the hours' UTC midpoints."""
    import time

    import xarray as xr

    from .sources.dynamical import open_dataset

    ds = open_dataset(NAME)
    glat, glon = ds["latitude"].values, ds["longitude"].values
    lon = ((st["longitude"].to_numpy() + 180) % 360) - 180
    li = np.abs(glat[:, None] - st["latitude"].to_numpy()[None]).argmin(0)
    lo = np.abs(glon[:, None] - lon[None]).argmin(0)
    cells = np.stack([[np.clip(li + i, 0, len(glat) - 1), (lo + j) % len(glon)] for i, j in OFFSETS], 1)  # (2, 9, n)
    flat = cells[0] * len(glon) + cells[1]
    uniq, inv = np.unique(flat, return_inverse=True)
    inv = inv.reshape(flat.shape).T  # (n, 9)
    sub = (ds[["precipitation_surface", "temperature_2m", "percent_frozen_precipitation_surface"]]
           .sel(init_time=np.datetime64(init)).isel(lead_time=slice(0, N_LEAD))
           .isel(latitude=xr.DataArray(uniq // len(glon), dims="pt"),
                 longitude=xr.DataArray(uniq % len(glon), dims="pt")))
    for attempt in range(retries):
        try:
            v = sub.load()
            break
        except Exception:  # flaky range reads
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    lead_h = np.asarray(pd.to_timedelta(v["lead_time"].values) / pd.Timedelta(hours=1), float)
    dt = np.diff(lead_h, prepend=0.0)
    keep = dt > 0
    arr = lambda n: v[n].transpose("pt", "lead_time").to_numpy().astype("float32")[:, keep]  # noqa: E731
    liq = np.nan_to_num(arr("precipitation_surface")).clip(min=0) * (dt[keep] * 3600).astype("float32")  # mm
    t = arr("temperature_2m")
    fz = arr("percent_frozen_precipitation_surface") / 100.0
    fz = np.where(np.isfinite(fz), fz, 1.0 / (1.0 + np.exp((t - 1.0) / 0.5)))
    fz = np.clip(np.nan_to_num(fz), 0, 1)
    snow = liq * fz * slr_2m(np.nan_to_num(t))
    mid = (pd.Timestamp(init) + pd.to_timedelta(lead_h[keep] - dt[keep] / 2, unit="h")).to_numpy()
    return snow[inv] / MM_PER_IN, liq[inv] / MM_PER_IN, (liq * fz)[inv] / MM_PER_IN, t[inv], mid


def _local_hours(mid: np.ndarray, st: pd.DataFrame, round_date: date) -> np.ndarray:
    """Hours from local midnight of the round date to each step's midpoint, (station, hour)."""
    tz = station_tz(st)
    out = np.empty((len(st), len(mid)), "float32")
    utc = pd.DatetimeIndex(mid).tz_localize("UTC")
    d0 = pd.Timestamp(round_date)
    for zone, idx in tz.groupby(tz).groups.items():
        rel = (utc.tz_convert(zone).tz_localize(None) - d0) / pd.Timedelta(hours=1)
        out[st.index.get_indexer(idx)] = np.asarray(rel, "float32")
    return out


def features(init: pd.Timestamp, st: pd.DataFrame, round_date: date) -> pd.DataFrame:
    """One run, one league's stations -> a row per station and horizon."""
    snow, liq, frz, t, mid = _read(init, st)
    rel = _local_hours(mid, st, round_date)[:, None, :]  # (n, 1, L)
    mean_series = snow.mean(1)  # (n, L): the neighbourhood's hourly snow
    r = rel[:, 0, :]
    rows = []
    for h in HORIZONS:
        win = (rel >= 0) & (rel < h)
        full = win[:, 0, :].sum(1) >= h - 1  # the run covers the whole window
        tot = (snow * win).sum(2)  # (n, 9)
        lq = (liq * win).sum(2)
        fr = (frz * win).sum(2)
        wet = np.take_along_axis(fr / np.clip(lq, 0.004, None), tot.argmax(1)[:, None], 1)[:, 0]
        tm = np.where(win, t, np.nan)
        w = win[:, 0, :]
        six = np.stack([(mean_series * (w & (r >= a) & (r < a + 6))).sum(1) for a in range(0, h - 5, 3)], 1)
        f = pd.DataFrame({
            "ctr": tot[:, 4], "max": tot.max(1), "mean": tot.mean(1), "min": tot.min(1), "sd": tot.std(1),
            "liq_max": lq.max(1), "frz_max": wet, "tmin": np.nanmin(np.nanmean(tm, 2), 1),
            "first12": (mean_series * (w & (r < 12))).sum(1),
            "last12": (mean_series * (w & (r >= h - 12))).sum(1),
            "pre12": (mean_series * ((r >= -12) & (r < 0))).sum(1),
            "post6": (mean_series * ((r >= h) & (r < h + 6))).sum(1),
            "peak6": six.max(1), "hours": ((mean_series > 0.01) & w).sum(1),
            # second set (2026-11-16), experimental until measured
            "pre6": (mean_series * ((r >= -6) & (r < 0))).sum(1),
            "pre24": (mean_series * ((r >= -24) & (r < 0))).sum(1),
            "post12": (mean_series * ((r >= h) & (r < h + 12))).sum(1),
            "first6": (mean_series * (w & (r < 6))).sum(1),
            "last6": (mean_series * (w & (r >= h - 6))).sum(1),
            "last24": (mean_series * (w & (r >= h - 24))).sum(1),
            "liq_pre12": (liq.mean(1) * ((r >= -12) & (r < 0))).sum(1),
            "rain": ((liq - frz).mean(1) * w).sum(1),
            "t_first12": np.nanmean(np.where(w & (r < 12), t[:, 4, :], np.nan), 1),
            "t_last12": np.nanmean(np.where(w & (r >= h - 12), t[:, 4, :], np.nan), 1),
        }, index=st.index).astype("float32")
        f[~full] = np.nan
        f.columns = [f"{PREFIX}_{c}" for c in f.columns]
        f["horizon_h"] = h
        rows.append(f.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["round_date"] = round_date
    return out.dropna(subset=[f"{PREFIX}_ctr"])


def for_round(league: str, round_date: date, st: pd.DataFrame | None = None, latest: bool = False) -> pd.DataFrame:
    """The features a forecast for this round may use; empty when the run is not in the archive
    (`latest`: fall back to the newest older run instead)."""
    from . import contract
    from .sources.dynamical import available_inits

    st = contract.stations(league) if st is None else st
    init = init_for_round(round_date, league, NAME)
    have = available_inits(NAME)
    if init not in have:
        if not latest:
            return pd.DataFrame()
        # live, before the cutoff's own run is out: the newest run there is (up to a day older;
        # windows it does not cover completely come back empty)
        older = have[(have <= init) & (have >= init - pd.Timedelta(hours=24))]
        if not len(older):
            return pd.DataFrame()
        init = older.max()
    return features(init, st, round_date)


FILES = f"{PREFIX}2"  # bumped when columns are added, so the harvest starts over


def asset(league: str, season: int) -> str:
    return f"{FILES}__{league}__{int(season)}.parquet"


TIMING = ("first12", "last12", "pre12", "post6", "peak6", "hours")


def join(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the harvested features (the `nbhd` release) onto a training table."""
    return join_files(df, FILES, PREFIX, "hourly-GFS")


def join_files(df: pd.DataFrame, files: str, prefix: str, label: str, key: str = "ctr") -> pd.DataFrame:
    from . import WORK, store

    dest = WORK / TAG
    store.pull(TAG, dest, f"{files}__*.parquet")
    df = df.drop(columns=[c for c in df.columns if c.startswith(f"{prefix}_")])
    parts = []
    for f in sorted(dest.glob(f"{files}__*.parquet")):
        g = pd.read_parquet(f)
        g["league"] = f.stem.split("__")[1]
        parts.append(g)
    if not parts:
        return df
    g = pd.concat(parts, ignore_index=True)
    g["round_date"] = pd.to_datetime(g["round_date"]).dt.date
    g = g.drop_duplicates(["league", "station_id", "round_date", "horizon_h"], keep="last")
    key = pd.to_datetime(df["round_date"]).dt.date
    out = df.assign(round_date=key).merge(g, on=["league", "station_id", "round_date", "horizon_h"], how="left")
    print(f"  {label}: {out[f'{prefix}_{key}'].notna().mean():.0%} of rows have the features", flush=True)
    return out
