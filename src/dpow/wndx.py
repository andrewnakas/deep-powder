"""Storm wind against the terrain: hourly-GFS flow and moisture during the snow.

How much of a storm a mountain station catches depends on the direction the
moist air comes from: upslope flow onto its side of the range wrings out
more, a shadowed site gets less than the 0.25 degree grid says. The daily
features carry amounts and temperatures but no wind. One hourly GFS run
(dynamical.org archive, no quota; same run rule as nbhd.py) gives the flow
during the hours it precipitates.

Columns, per station, round and horizon, each weighted by the station
cell's hourly precipitation inside the window (plain window mean when dry):
  wndx_u, wndx_v, wndx_spd   100 m wind (m/s, toward east / north)
  wndx_up                    flow up the grid's own terrain slope (u.grad z, mm/s; z from the
                             3x3 cells' surface pressure)
  wndx_up_asp                flow onto the station's local slope face (DEM aspect and slope, m/s)
  wndx_pw, wndx_rh           precipitable water (kg/m2), 2 m relative humidity (%)
  wndx_t80                   80 m temperature (C)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import HORIZONS, NEW_LEAGUES
from .ens_features import init_for_round
from .nbhd import OFFSETS, _local_hours

NAME = "gfs"
PREFIX = "wndx"
TAG = "nbhd"
FILES = "wndx"
LEAGUES = ("stations", "observers", "alps", *NEW_LEAGUES)  # GFS is global
N_LEAD = 105
VARS = ("precipitation_surface", "wind_u_100m", "wind_v_100m", "pressure_surface",
        "precipitable_water_atmosphere", "relative_humidity_2m", "temperature_80m")
SCALE_H = 8000.0  # m: z ~ -H ln p over a few cells


def _read(init: pd.Timestamp, st: pd.DataFrame, retries: int = 5):
    """-> dict of (station, hour) arrays for the own cell, terrain gradient (dz/dx, dz/dy) per
    station, step lengths and the steps' UTC midpoints."""
    import time

    import xarray as xr

    from .sources.dynamical import open_dataset

    ds = open_dataset(NAME)
    glat, glon = ds["latitude"].values, ds["longitude"].values
    lat = st["latitude"].to_numpy()
    lon = ((st["longitude"].to_numpy() + 180) % 360) - 180
    li = np.abs(glat[:, None] - lat[None]).argmin(0)
    lo = np.abs(glon[:, None] - lon[None]).argmin(0)
    cells = np.stack([[np.clip(li + i, 1, len(glat) - 2) if i else li, (lo + j) % len(glon)] for i, j in OFFSETS], 1)
    flat = cells[0] * len(glon) + cells[1]
    uniq, inv = np.unique(flat, return_inverse=True)
    inv = inv.reshape(flat.shape).T  # (n, 9)
    sub = (ds[list(VARS)].sel(init_time=np.datetime64(init)).isel(lead_time=slice(0, N_LEAD))
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

    def arr(n):
        return v[n].transpose("pt", "lead_time").to_numpy().astype("float32")[:, keep]

    own = {n: arr(n)[inv[:, 4]] for n in VARS if n != "pressure_surface"}
    own["precipitation_surface"] = np.nan_to_num(own["precipitation_surface"]).clip(min=0) * dt[keep]
    # terrain height of each of the 9 cells from the time-mean surface pressure
    z = -SCALE_H * np.log(np.nanmean(arr("pressure_surface"), 1).clip(min=1.0))[inv]  # (n, 9)
    # OFFSETS index: (i, j) -> 3*(i+1) + (j+1); i steps the latitude index, j the longitude index
    up_i, dn_i = 3 * 0 + 1, 3 * 2 + 1  # (-1, 0), (1, 0)
    lf_j, rt_j = 3 * 1 + 0, 3 * 1 + 2  # (0, -1), (0, 1)
    dy = (glat[np.clip(li - 1, 0, len(glat) - 1)] - glat[np.clip(li + 1, 0, len(glat) - 1)]) * 111_000.0
    dx = 2 * (glon[1] - glon[0]) * 111_000.0 * np.cos(np.radians(lat))
    dzdy = (z[:, up_i] - z[:, dn_i]) / np.where(dy == 0, np.nan, dy)
    dzdx = (z[:, rt_j] - z[:, lf_j]) / dx
    mid = (pd.Timestamp(init) + pd.to_timedelta(lead_h[keep] - dt[keep] / 2, unit="h")).to_numpy()
    return own, dzdx.astype("float32"), dzdy.astype("float32"), dt[keep], mid


def _terrain(st: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> sin and cos of the slope's aspect, tan of its slope, per station (NaN when unknown)."""
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "data" / "terrain.parquet"
    if not p.exists():
        nan = np.full(len(st), np.nan)
        return nan, nan, nan
    t = pd.read_parquet(p, columns=["station_id", "aspect_deg", "slope_deg"]).set_index("station_id")
    t = t[~t.index.duplicated()].reindex(st.index)
    a, s = np.radians(t["aspect_deg"].to_numpy(float)), np.radians(t["slope_deg"].to_numpy(float))
    return np.sin(a), np.cos(a), np.tan(s)


def features(init: pd.Timestamp, st: pd.DataFrame, round_date: date) -> pd.DataFrame:
    own, dzdx, dzdy, dt, mid = _read(init, st)
    r = _local_hours(mid, st, round_date)  # (n, L)
    asin, acos, tan = _terrain(st)
    u, v = own["wind_u_100m"], own["wind_v_100m"]
    up = (u * dzdx[:, None] + v * dzdy[:, None]) * 1000.0  # mm/s
    # aspect is the direction the slope faces (downhill); flow toward the face from outside is upslope
    up_asp = -(u * asin[:, None] + v * acos[:, None]) * tan[:, None]
    fields = {"u": u, "v": v, "spd": np.hypot(u, v), "up": up, "up_asp": up_asp,
              "pw": own["precipitable_water_atmosphere"], "rh": own["relative_humidity_2m"],
              "t80": own["temperature_80m"]}
    p = own["precipitation_surface"]
    rows = []
    for h in HORIZONS:
        w = (r >= 0) & (r < h)
        full = (w * dt).sum(1) >= h - 1
        wt = p * w
        tot = wt.sum(1, keepdims=True)
        wt = np.where(tot > 0.1, wt / np.where(tot > 0, tot, 1), w / np.clip(w.sum(1, keepdims=True), 1, None))
        f = pd.DataFrame({k: np.nansum(np.nan_to_num(a) * wt, 1) for k, a in fields.items()},
                         index=st.index).astype("float32")
        f[~full] = np.nan
        f.columns = [f"{PREFIX}_{c}" for c in f.columns]
        f["horizon_h"] = h
        rows.append(f.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["round_date"] = round_date
    return out.dropna(subset=[f"{PREFIX}_u"])


def for_round(league: str, round_date: date, st: pd.DataFrame | None = None, latest: bool = False) -> pd.DataFrame:
    from . import contract
    from .sources.dynamical import available_inits

    if league not in LEAGUES:
        return pd.DataFrame()
    st = contract.stations(league) if st is None else st
    init = init_for_round(round_date, league, NAME)
    have = available_inits(NAME)
    if init not in have:
        older = have[(have <= init) & (have >= init - pd.Timedelta(hours=24))] if latest else []
        if not len(older):
            return pd.DataFrame()
        init = older.max()
    return features(init, st, round_date)


def asset(league: str, season: int) -> str:
    return f"{FILES}__{league}__{int(season)}.parquet"


def join(df: pd.DataFrame) -> pd.DataFrame:
    from .nbhd import join_files

    return join_files(df, FILES, PREFIX, "storm wind", key="u")
