"""Hourly HRRR (3 km, CONUS, 48 h runs) at each station: 24 h snow and its timing.

The 0.25 degree GFS sees a mountain range as a few cells; HRRR resolves the
ridge. The ensemble features take HRRR as daily sums at one cell. These add
what that hides, for the 24 h window only (a usable run does not reach the
end of the 48 h window): its own snowfall field and its precipitation-type
snow, at the station's cell and as the snowiest of the 3x3 cells around it,
and when inside (and just outside) the window the snow falls.

Same run rule as everything else (ens_features.init_for_round), so training
and live see the same lead time. CONUS stations of stations/observers only.

Columns (inches of snow), per station and round, horizon 24:
  hrrx_ctr, hrrx_max          precipitation-type snow, own cell / snowiest of 3x3
  hrrx_nat, hrrx_nat_max      HRRR's own snowfall field, own cell / snowiest of 3x3
  hrrx_first12, hrrx_last12   first and last 12 local hours (own cell, precip-type)
  hrrx_pre12, hrrx_post6      just outside the window
  hrrx_peak3, hrrx_hours      heaviest 3 h, hours with snow
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .ens_features import MM_PER_IN, init_for_round, slr_2m
from .nbhd import _local_hours

NAME = "hrrr"
PREFIX = "hrrx"
TAG = "nbhd"  # same release as the GFS features
FILES = "hrrx"
LEAGUES = ("stations", "observers")
OFFSETS = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]  # index 4 is the station's own cell
H = 24


def _read(init: pd.Timestamp, st: pd.DataFrame, retries: int = 5):
    import time

    import xarray as xr

    from .sources.dynamical import _nearest_yx, open_dataset

    ds = open_dataset(NAME)
    ny, nx = ds["latitude"].shape
    lon = ((st["longitude"].to_numpy() + 180) % 360) - 180
    yi, xi = _nearest_yx(NAME, st["latitude"].to_numpy(), lon)
    cells = np.stack([[np.clip(yi + i, 0, ny - 1), np.clip(xi + j, 0, nx - 1)] for i, j in OFFSETS], 1)
    flat = cells[0] * nx + cells[1]
    uniq, inv = np.unique(flat, return_inverse=True)
    inv = inv.reshape(flat.shape).T  # (n, 9)
    sub = (ds[["precipitation_surface", "temperature_2m", "percent_frozen_precipitation_surface", "snowfall_surface"]]
           .sel(init_time=np.datetime64(init))
           .isel(y=xr.DataArray(uniq // nx, dims="pt"), x=xr.DataArray(uniq % nx, dims="pt")))
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
    sec = (dt[keep] * 3600).astype("float32")
    arr = lambda n: v[n].transpose("pt", "lead_time").to_numpy().astype("float32")[:, keep]  # noqa: E731
    liq = np.nan_to_num(arr("precipitation_surface")).clip(min=0) * sec
    t = arr("temperature_2m")
    fz = arr("percent_frozen_precipitation_surface") / 100.0
    fz = np.clip(np.nan_to_num(np.where(np.isfinite(fz), fz, 1.0 / (1.0 + np.exp((t - 1.0) / 0.5)))), 0, 1)
    snow = liq * fz * slr_2m(np.nan_to_num(t)) / MM_PER_IN
    nat = np.nan_to_num(arr("snowfall_surface")).clip(min=0) * sec * 1000 / MM_PER_IN
    mid = (pd.Timestamp(init) + pd.to_timedelta(lead_h[keep] - dt[keep] / 2, unit="h")).to_numpy()
    return snow[inv], nat[inv], mid


def features(init: pd.Timestamp, st: pd.DataFrame, round_date: date) -> pd.DataFrame:
    snow, nat, mid = _read(init, st)
    r = _local_hours(mid, st, round_date)  # (n, L)
    w = (r >= 0) & (r < H)
    full = w.sum(1) >= H - 1
    tot = (snow * w[:, None, :]).sum(2)
    ntot = (nat * w[:, None, :]).sum(2)
    c = snow[:, 4, :]
    three = np.stack([(c * (w & (r >= a) & (r < a + 3))).sum(1) for a in range(0, H - 2)], 1)
    f = pd.DataFrame({
        "ctr": tot[:, 4], "max": tot.max(1), "nat": ntot[:, 4], "nat_max": ntot.max(1),
        "first12": (c * (w & (r < 12))).sum(1), "last12": (c * (w & (r >= H - 12))).sum(1),
        "pre12": (c * ((r >= -12) & (r < 0))).sum(1), "post6": (c * ((r >= H) & (r < H + 6))).sum(1),
        "peak3": three.max(1), "hours": ((c > 0.01) & w).sum(1),
    }, index=st.index).astype("float32")
    f[~full] = np.nan
    f.columns = [f"{PREFIX}_{k}" for k in f.columns]
    f["horizon_h"] = H
    out = f.reset_index()
    out["round_date"] = round_date
    return out.dropna(subset=[f"{PREFIX}_ctr"])


def for_round(league: str, round_date: date, st: pd.DataFrame | None = None, latest: bool = False) -> pd.DataFrame:
    from . import contract
    from .sources.dynamical import available_inits, in_conus

    if league not in LEAGUES:
        return pd.DataFrame()
    st = contract.stations(league) if st is None else st
    st = st[in_conus(st["latitude"], st["longitude"])]
    if st.empty:
        return pd.DataFrame()
    init = init_for_round(round_date, league, NAME)
    have = available_inits(NAME)
    if init not in have:
        older = have[(have <= init) & (have >= init - pd.Timedelta(hours=12))] if latest else []
        if not len(older):
            return pd.DataFrame()
        init = older.max()
    return features(init, st, round_date)


def asset(league: str, season: int) -> str:
    return f"{FILES}__{league}__{int(season)}.parquet"


def join(df: pd.DataFrame) -> pd.DataFrame:
    from .nbhd import join_files

    return join_files(df, FILES, PREFIX, "hourly-HRRR")
