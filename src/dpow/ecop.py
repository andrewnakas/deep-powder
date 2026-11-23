"""ECMWF IFS (the deterministic HRES run) at every station, read straight
from ECMWF's open data on AWS instead of through Open-Meteo's `ecmwf_ifs025`
(which is built from these same files).

Source: s3://ecmwf-forecasts (eu-central-1, anonymous), from 2023-01-18.
Licence: ECMWF open data, CC BY 4.0 (Copyright ECMWF). The archive is kept,
not rolling. Layout by date (00/12 UTC runs, stream `oper`):
- to 2024-01: `<d>/<HH>z/0p4-beta/oper/` (0.4 deg, no dew point)
- 2024-02:    `<d>/<HH>z/0p25/oper/`
- after:      `<d>/<HH>z/ifs/0p25/oper/`
Each step is one GRIB2 file (~120 MB at 0.25 deg) with a JSON-lines
`.index` of message offsets, so a field is one range read: total
precipitation, 2 m temperature and dew point, ~2 MB per step.

A unit = (run hour, month), every station of every league, one row per
station and run, in the Met Office harvest's layout (ukmo.py), so the same
`hourly`/`windows` code builds the features:
- `pr_001`..`pr_096`: hourly precipitation (mm), each 3-hour step's total
  spread evenly over its 3 hours
- `t2_00`..`t2_96`, `td_00`..`td_96`: 2 m temperature and dew point (C), every 3 h
Assets of the `ecop` release: `ecop__<HH>z__<YYYY-MM>.parquet`.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from . import WORK, store
from .ukmo import T_LEADS, stations

BUCKET = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
TAG = "ecop"
DIR = WORK / "ecop"
MODEL = "ecmwf_op"
RUN_HOURS = (12, 0)
FIRST = pd.Timestamp("2023-01-18")
STEPS = T_LEADS  # 0, 3, ..., 96
PARAMS = {"tp": "tp", "2t": "t2", "2d": "td"}
LAYOUTS = ("ifs/0p25/oper", "0p25/oper", "0p4-beta/oper", "ifs/0p4-beta/oper")
UA = {"User-Agent": "deep-powder (github.com/andrewnakas/deep-powder)"}
RETRIES = 7
_local = threading.local()


def unit_name(hour: int, month: pd.Period) -> str:
    return f"{TAG}__{hour:02d}z__{month}.parquet"


def _session() -> requests.Session:
    if getattr(_local, "s", None) is None:
        _local.s = requests.Session()
        _local.s.headers.update(UA)
    return _local.s


def _get(url: str, headers: dict | None = None) -> requests.Response | None:
    """None on 404/403. S3 answers a too-fast client with 503 SlowDown: back off
    (up to ~4 min in all) rather than lose the run."""
    for attempt in range(RETRIES):
        try:
            r = _session().get(url, headers=headers, timeout=(30, 90))
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            _local.s = None
            if attempt == RETRIES - 1:
                raise
            slow = getattr(getattr(e, "response", None), "status_code", None) in (429, 500, 503)
            time.sleep(min(60, (8 if slow else 2) * 2 ** attempt))
    return None


def file_url(run: pd.Timestamp, layout: str, step: int, ext: str = "grib2") -> str:
    return f"{BUCKET}/{run:%Y%m%d}/{run:%H}z/{layout}/{run:%Y%m%d%H}0000-{step}h-oper-fc.{ext}"


def layout_for(run: pd.Timestamp) -> str | None:
    """The run's directory, or None when the run is not in the bucket."""
    for lay in LAYOUTS:
        if _get(file_url(run, lay, 0, "index")) is not None:
            return lay
    return None


def messages(index: str) -> dict[str, tuple[int, int]]:
    """param -> (offset, length) of the surface fields we keep."""
    out = {}
    for ln in index.splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        if d.get("levtype") == "sfc" and d.get("param") in PARAMS:
            out[d["param"]] = (int(d["_offset"]), int(d["_length"]))
    return out


def decode(msg: bytes) -> tuple[np.ndarray, tuple]:
    """Values and the regular lat/lon grid they are on."""
    import eccodes

    gid = eccodes.codes_new_from_message(msg)
    try:
        vals = np.asarray(eccodes.codes_get_values(gid), "float32")
        g = tuple(eccodes.codes_get(gid, k) for k in (
            "Ni", "Nj", "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
            "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees", "jScansPositively"))
    finally:
        eccodes.codes_release(gid)
    return vals, g


def cell_index(grid: tuple, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Flat index of each station's nearest grid point."""
    ni, nj, lat0, lon0, di, dj, jpos = grid
    j = np.rint((lat - lat0) / dj) if jpos else np.rint((lat0 - lat) / dj)
    i = np.rint(((lon - lon0) % 360.0) / di) % ni
    return (np.clip(j, 0, nj - 1).astype(np.int64) * ni + i.astype(np.int64))


def _read_step(run: pd.Timestamp, layout: str, step: int, lat, lon) -> dict[str, np.ndarray]:
    idx = _get(file_url(run, layout, step, "index"))
    if idx is None:
        return {}
    url = file_url(run, layout, step)
    out = {}
    for p, (off, n) in messages(idx.text).items():
        r = _get(url, {"Range": f"bytes={off}-{off + n - 1}"})
        if r is None or len(r.content) != n:
            continue
        vals, grid = decode(r.content)
        out[PARAMS[p]] = vals[cell_index(grid, lat, lon)]
    return out


def harvest_run(run: pd.Timestamp, st: pd.DataFrame, pool: ThreadPoolExecutor) -> pd.DataFrame | None:
    layout = layout_for(run)
    if layout is None:
        return None
    lat, lon = st["latitude"].to_numpy(float), st["longitude"].to_numpy(float)
    got = dict(zip(STEPS, pool.map(lambda s: _read_step(run, layout, s, lat, lon), STEPS)))
    n = len(st)
    nan = np.full(n, np.nan, np.float32)
    cols = {}
    prev = got[STEPS[0]].get("tp", np.zeros(n, np.float32))  # tp accumulates from step 0
    for a, b in zip(STEPS[:-1], STEPS[1:]):
        cur = got[b].get("tp")
        per_h = nan if cur is None or prev is None else np.maximum(cur - prev, 0) * 1000 / (b - a)  # m -> mm/h
        for h in range(a + 1, b + 1):
            cols[f"pr_{h:03d}"] = per_h
        prev = cur
    for v in ("t2", "td"):
        for s in STEPS:
            x = got[s].get(v)
            cols[f"{v}_{s:02d}"] = nan if x is None else x - 273.15
    from .ukmo import columns

    df = pd.DataFrame({c: np.round(cols[c], 2).astype(np.float32) for c in columns()})
    df.insert(0, "run", run)
    df.insert(0, "station_id", st.index.values)
    return df


def runs_in(hour: int, month: pd.Period, now: pd.Timestamp) -> list[pd.Timestamp]:
    days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
    return [r for r in (d + pd.Timedelta(hours=hour) for d in days) if FIRST <= r <= now - pd.Timedelta(hours=8)]


def units(now: pd.Timestamp, hours: tuple[int, ...] = RUN_HOURS) -> list[tuple[int, pd.Period]]:
    """Complete months; northern winter months first (most leagues), then the
    rest, oldest first within each."""
    last = (now - pd.Timedelta(days=1)).to_period("M") - 1
    months = list(pd.period_range(FIRST.to_period("M"), last, freq="M"))
    nh = [m for m in months if m.month in (11, 12, 1, 2, 3, 4)]
    rest = [m for m in months if m not in nh]
    return [(h, m) for h in hours for m in nh + rest]


def run(minutes: float, workers: int = 6, hours: tuple[int, ...] = RUN_HOURS) -> dict:
    now = pd.Timestamp.utcnow().tz_localize(None)
    deadline = time.time() + minutes * 60
    have = store.remote_assets(TAG) | {p.name for p in DIR.glob(f"{TAG}__*.parquet")}
    todo = [u for u in units(now, hours) if unit_name(*u) not in have]
    print(f"ecop: {len(todo)} units to go, {minutes:.0f} min budget, {workers} threads", flush=True)
    if not todo:
        return {"done": 0, "remaining": 0}
    DIR.mkdir(parents=True, exist_ok=True)
    st = stations()
    print(f"ecop: {len(st)} stations", flush=True)
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for hour, month in todo:
            t0 = time.time()
            frames, missing = [], 0
            for r in runs_in(hour, month, now):
                if time.time() > deadline:
                    print("  time budget reached", flush=True)
                    return {"done": done, "remaining": len(todo) - done}
                t1 = time.time()
                df = None
                for attempt in range(3):  # a throttled run waits and tries again
                    try:
                        df = harvest_run(r, st, pool)
                        break
                    except Exception as e:
                        print(f"  {r}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                        time.sleep(120 * (attempt + 1))
                print(f"    {r:%Y-%m-%d %HZ}: {'missing' if df is None else 'ok'}, {time.time() - t1:.0f}s", flush=True)
                if df is None:
                    missing += 1
                else:
                    frames.append(df)
            if not frames:
                print(f"  {unit_name(hour, month)}: no runs in the bucket", flush=True)
                continue
            p = DIR / unit_name(hour, month)
            pd.concat(frames, ignore_index=True).to_parquet(p, index=False, compression="zstd")
            store.push(TAG, [p])
            done += 1
            print(f"  {p.name}: {len(frames)} runs ({missing} missing), {p.stat().st_size / 1e6:.0f} MB, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return {"done": done, "remaining": len(todo) - done}


def pull() -> None:
    store.pull(TAG, DIR, f"{TAG}__*.parquet", overwrite=False)


def windows(hour: int = 12) -> pd.DataFrame:
    """Per (station_id, round_date, horizon_h): `<var>__ecmwf_op` columns,
    built like the Open-Meteo models' (see ukmo.windows)."""
    from .ukmo import windows_from

    return windows_from(sorted(DIR.glob(f"{TAG}__{hour:02d}z__*.parquet")), MODEL)
