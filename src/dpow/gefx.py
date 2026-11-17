"""GEFS (31 members, 3-hourly) timing: when inside the window the snow falls,
and how sure the ensemble is about it.

The hourly-GFS timing columns (nbhd.py) were the largest single gain any
input has given (+1.5 CV in all three entries). They come from one
deterministic run. GEFS gives the same timing from 31 members: the
ensemble's mean split of the window, its spread, and the share of members
that put real snow just before or just after the window (the cases an
observer's fixed reading hour moves into the neighbouring day).

Same run rule as the daily ensemble features (ens_features.init_for_round),
so training and live see the same lead time. Station's own cell. Archive
from 2020-10; reads are small (chunks hold 64 leads x 17 x 16 cells, every
member).

Columns (inches of snow; ensemble mean unless noted), per station, round, horizon:
  gefx_first12, gefx_last12      first and last 12 local hours of the window
  gefx_pre12, gefx_post6         just before / just after it
  gefx_peak6, gefx_hours         heaviest 6 h of the member mean; hours with snow (mean)
  gefx_first_sd, gefx_last_sd    member spread of first12 / last12
  gefx_p_pre, gefx_p_post        share of members with >= 0.5 in. in pre12 / post6
  gefx_p_edge                    share of members whose window snow falls mostly in its last 12 h
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import HORIZONS
from .ens_features import MM_PER_IN, init_for_round, slr_2m
from .nbhd import _local_hours

NAME = "gefs"
PREFIX = "gefx"
TAG = "nbhd"
FILES = "gefx"
LEAGUES = ("stations", "observers", "alps")
N_LEAD = 41  # 0..120 h at 3 h steps: covers the 72 h window from the run the rule picks


def _read(init: pd.Timestamp, st: pd.DataFrame, retries: int = 5):
    """-> snow (in) (station, member, step), step length (h) and the steps' UTC midpoints."""
    import time

    import xarray as xr

    from .sources.dynamical import open_dataset

    ds = open_dataset(NAME)
    glat, glon = ds["latitude"].values, ds["longitude"].values
    lon = ((st["longitude"].to_numpy() + 180) % 360) - 180
    li = np.abs(glat[:, None] - st["latitude"].to_numpy()[None]).argmin(0)
    lo = np.abs(glon[:, None] - lon[None]).argmin(0)
    flat = li * len(glon) + lo
    uniq, inv = np.unique(flat, return_inverse=True)
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
    def arr(n):
        return v[n].transpose("pt", "ensemble_member", "lead_time").to_numpy().astype("float32")[..., keep]
    liq = np.nan_to_num(arr("precipitation_surface")).clip(min=0) * (dt[keep] * 3600).astype("float32")  # mm
    t = arr("temperature_2m")
    fz = arr("percent_frozen_precipitation_surface") / 100.0
    fz = np.clip(np.nan_to_num(np.where(np.isfinite(fz), fz, 1.0 / (1.0 + np.exp((t - 1.0) / 0.5)))), 0, 1)
    snow = liq * fz * slr_2m(np.nan_to_num(t)) / MM_PER_IN
    mid = (pd.Timestamp(init) + pd.to_timedelta(lead_h[keep] - dt[keep] / 2, unit="h")).to_numpy()
    return snow[inv], dt[keep], mid


def features(init: pd.Timestamp, st: pd.DataFrame, round_date: date) -> pd.DataFrame:
    snow, dt, mid = _read(init, st)  # (n, m, L)
    r = _local_hours(mid, st, round_date)[:, None, :]  # (n, 1, L)
    rows = []
    for h in HORIZONS:
        w = (r >= 0) & (r < h)
        full = (w[:, 0, :] * dt).sum(1) >= h - 3  # the run covers the whole window
        tot = (snow * w).sum(2)
        first = (snow * (w & (r < 12))).sum(2)
        last = (snow * (w & (r >= h - 12))).sum(2)
        pre = (snow * ((r >= -12) & (r < 0))).sum(2)
        post = (snow * ((r >= h) & (r < h + 6))).sum(2)
        mean = snow.mean(1)  # (n, L)
        r0 = r[:, 0, :]
        six = np.stack([(mean * (w[:, 0, :] & (r0 >= a) & (r0 < a + 6))).sum(1) for a in range(0, h - 5, 3)], 1)
        f = pd.DataFrame({
            "first12": first.mean(1), "last12": last.mean(1), "pre12": pre.mean(1), "post6": post.mean(1),
            "peak6": six.max(1), "hours": (((snow > 0.01) & w) * dt).sum(2).mean(1),
            "first_sd": first.std(1), "last_sd": last.std(1),
            "p_pre": (pre >= 0.5).mean(1), "p_post": (post >= 0.5).mean(1),
            "p_edge": ((tot > 0.1) & (last > 0.5 * tot)).mean(1),
        }, index=st.index).astype("float32")
        f[~full] = np.nan
        f.columns = [f"{PREFIX}_{c}" for c in f.columns]
        f["horizon_h"] = h
        rows.append(f.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["round_date"] = round_date
    return out.dropna(subset=[f"{PREFIX}_first12"])


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

    return join_files(df, FILES, PREFIX, "GEFS timing", key="first12")
