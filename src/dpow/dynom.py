"""Open-Meteo-style model columns for every station from dynamical.org's
deterministic archives: GFS (from 2021-05) and HRRR (from 2018-08, CONUS).

Open-Meteo's `gfs_seamless` and `ncep_hrrr_conus` columns are on only the
featured stations and a sample (11-26% of table rows). The same runs are in
dynamical.org's Icechunk stores for every grid point, so the 12Z run of each
day is read at every station and laid out like the Met Office harvest
(ukmo.py: `pr_001`..`pr_096` hourly precipitation in mm, `t2_00`..`t2_96`
2 m temperature in C every 3 h), and `ukmo.windows_from` builds the same
round windows as the Open-Meteo models'.

A unit = (dataset, month): `dynom__<ds>__12z__<YYYY-MM>.parquet` assets of
the `dynom` release. Resumable; `part`/`parts` split the units between jobs.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import WORK, store
from .ukmo import MAX_LEAD, T_LEADS, columns, stations, windows_from

TAG = "dynom"
DIR = WORK / "dynom"
RUN_HOUR = 12
MODELS = {"gfs": "gfs_dyn", "hrrr": "hrrr_dyn"}  # dataset -> model name in the columns
OM_TWIN = {"gfs": "gfs_seamless", "hrrr": "ncep_hrrr_conus"}  # the Open-Meteo model it stands in for


def unit_name(ds: str, month: pd.Period) -> str:
    return f"{TAG}__{ds}__{RUN_HOUR:02d}z__{month}.parquet"


def wide(long: pd.DataFrame, run: pd.Timestamp, st: pd.DataFrame) -> pd.DataFrame:
    """dynamical.extract's long frame (one run) -> one row per station in the
    ukmo layout. A step's precipitation is spread evenly over its hours."""
    n = len(st)
    pos = pd.Series(np.arange(n), index=st.index)
    d = long[long["member"] == long["member"].min()]
    lead = np.rint((pd.to_datetime(d["valid"]) - run) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    dt = np.rint(d["dt_h"].to_numpy()).astype(int)
    i = pos.reindex(d["station_id"]).to_numpy()
    ok = ~np.isnan(i) if i.dtype.kind == "f" else np.ones(len(i), bool)
    i, lead, dt = i[ok].astype(int), lead[ok], dt[ok]
    p = d["p"].to_numpy(np.float32)[ok]
    t = d["t2m"].to_numpy(np.float32)[ok]
    P = np.full((n, MAX_LEAD), np.nan, np.float32)
    for k in range(int(dt.max()) if len(dt) else 0):  # hour k before the step's end
        h = lead - k
        m = (k < dt) & (h >= 1) & (h <= MAX_LEAD)
        P[i[m], h[m] - 1] = p[m] / dt[m]
    T = np.full((n, len(T_LEADS)), np.nan, np.float32)
    tl = {h: j for j, h in enumerate(T_LEADS)}
    j = np.array([tl.get(x, -1) for x in lead])
    m = j >= 0
    T[i[m], j[m]] = t[m]
    cols = {f"pr_{h:03d}": P[:, h - 1] for h in range(1, MAX_LEAD + 1)}
    cols |= {f"t2_{h:02d}": T[:, j] for j, h in enumerate(T_LEADS)}
    cols |= {f"td_{h:02d}": np.full(n, np.nan, np.float32) for h in T_LEADS}
    df = pd.DataFrame({c: np.round(cols[c], 2).astype(np.float32) for c in columns()})
    df.insert(0, "run", run)
    df.insert(0, "station_id", st.index.values)
    return df


def harvest_run(ds: str, run: pd.Timestamp, st: pd.DataFrame) -> pd.DataFrame | None:
    from .sources.dynamical import DATASETS, extract, in_conus

    if DATASETS[ds].conus_only:
        st = st[in_conus(st["latitude"], st["longitude"])]
    try:
        long = extract(ds, run, st)
    except KeyError:  # run missing from the archive
        return None
    return None if long.empty else wide(long, run, st)


def units(datasets: list[str], end: pd.Timestamp) -> list[tuple[str, pd.Period]]:
    """Complete months, newest first (the test seasons first)."""
    from .sources.dynamical import DATASETS

    last = (end - pd.Timedelta(days=1)).to_period("M") - 1
    out = []
    for ds in datasets:
        months = pd.period_range(pd.Timestamp(DATASETS[ds].first_init).to_period("M"), last, freq="M")
        out += [(ds, m) for m in months]
    return sorted(out, key=lambda u: (-u[1].ordinal, u[0]))


def run(datasets: list[str], minutes: float, part: int = 0, parts: int = 1) -> dict:
    now = pd.Timestamp.utcnow().tz_localize(None)
    deadline = time.time() + minutes * 60
    have = store.remote_assets(TAG) | {p.name for p in DIR.glob(f"{TAG}__*.parquet")}
    todo = [u for k, u in enumerate(units(datasets, now)) if k % parts == part and unit_name(*u) not in have]
    print(f"dynom: {len(todo)} units to go ({', '.join(datasets)}), {minutes:.0f} min budget", flush=True)
    if not todo:
        return {"done": 0, "remaining": 0}
    DIR.mkdir(parents=True, exist_ok=True)
    st = stations()
    done = 0
    for ds, month in todo:
        t0 = time.time()
        frames, missing = [], 0
        days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
        for r in (d + pd.Timedelta(hours=RUN_HOUR) for d in days):
            if time.time() > deadline:
                print("  time budget reached", flush=True)
                return {"done": done, "remaining": len(todo) - done}
            try:
                df = harvest_run(ds, r, st)
            except Exception as e:  # keep the unit; the run stays missing
                print(f"  {ds} {r}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                df = None
            if df is None:
                missing += 1
            else:
                frames.append(df)
        if not frames:
            print(f"  {unit_name(ds, month)}: no runs", flush=True)
            continue
        p = DIR / unit_name(ds, month)
        pd.concat(frames, ignore_index=True).to_parquet(p, index=False, compression="zstd")
        store.push(TAG, [p])
        done += 1
        print(f"  {p.name}: {len(frames)} runs ({missing} missing), {p.stat().st_size / 1e6:.0f} MB, "
              f"{time.time() - t0:.0f}s", flush=True)
    return {"done": done, "remaining": len(todo) - done}


def pull(ds: str | None = None) -> None:
    store.pull(TAG, DIR, f"{TAG}__{ds or '*'}__{RUN_HOUR:02d}z__*.parquet", overwrite=False)


def windows(ds: str) -> pd.DataFrame:
    """`<var>__<gfs_dyn|hrrr_dyn>` round windows (see ukmo.windows_from)."""
    return windows_from(sorted(DIR.glob(f"{TAG}__{ds}__{RUN_HOUR:02d}z__*.parquet")), MODELS[ds])
