"""The Met Office global model (Unified Model, ~10 km) at every station,
read straight from the Met Office's open-data bucket on AWS instead of
through Open-Meteo's `ukmo_seamless`.

Source: s3://met-office-atmospheric-model-data/global-deterministic-10km
(eu-west-2, anonymous). Licence: British Crown copyright, the Met Office,
CC BY-SA 4.0. The bucket is a 2-year ROLLING archive: the oldest day is
deleted as each new one arrives, so the harvest runs oldest month first.

Each file is one variable at one lead of one run: NetCDF4, gzip, 128x128
chunks. Only the chunks that hold stations are read, using h5py's chunk
index and HTTP range requests: about 0.5 MB of a 4 MB file. h5py serialises
its calls behind one lock, so the files of a run are read by a process pool.

A unit = (run hour, month): every run at that hour in the month, every
station of every league, one row per station and run:
- `pr_001`..`pr_096`: hourly precipitation (mm) for leads 1-96 h; past
  48 h the model writes 3-hour totals, spread evenly over their 3 hours
- `t2_00`..`t2_96`, `td_00`..`td_96`: screen temperature and dew point (C),
  every 3 h
The bucket has no orography file. Downstream, 2 m temperature is moved to
the station's height with a standard lapse rate over `elev_delta_14km`
(the station's height above the mean terrain of a model-cell-sized box, in
data/terrain.parquet). Pressure-level fields are left out: their files are
270 MB with a 9,900-chunk index (~70 round trips per file just to find the
chunks), and dynamical.org's t850 covers the free atmosphere already.
Assets of the `ukmo` release: `ukmo__<HH>z__<YYYY-MM>.parquet`.
"""

from __future__ import annotations

import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import numpy as np
import pandas as pd

from . import WORK, store

BUCKET = "https://met-office-atmospheric-model-data.s3.eu-west-2.amazonaws.com"
PREFIX = "global-deterministic-10km"
TAG = "ukmo"
DIR = WORK / "ukmo"
LEAGUES = ("stations", "observers", "alps", "resorts-na", "resorts", "era5")
RUN_HOURS = (12, 0)  # 12Z first: the run every league's cutoff sees
FIRST = pd.Timestamp("2024-11-21")  # start of the rolling archive when this was written
MAX_LEAD = 96
P1_LEADS = tuple(range(1, 49))  # hourly precipitation files
P3_LEADS = tuple(range(51, MAX_LEAD + 1, 3))  # 3-hour totals
T_LEADS = tuple(range(0, MAX_LEAD + 1, 3))
LEVELS = (850, 700, 600)  # for read_points on pressure-level files (not harvested)
CHUNK = 128
VARS = {
    "pr1": "precipitation_accumulation-PT01H",
    "pr3": "precipitation_accumulation-PT03H",
    "t2": "temperature_at_screen_level",
    "td": "temperature_of_dew_point_at_screen_level",
    "t": "temperature_on_pressure_levels",
    "z": "height_ASL_on_pressure_levels",
}


def file_url(run: pd.Timestamp, lead: int, var: str) -> str:
    valid = run + pd.Timedelta(hours=lead)
    return f"{BUCKET}/{PREFIX}/{run:%Y%m%dT%H%MZ}/{valid:%Y%m%dT%H%MZ}-PT{lead:04d}H00M-{VARS[var]}.nc"


def unit_name(hour: int, month: pd.Period) -> str:
    return f"{TAG}__{hour:02d}z__{month}.parquet"


def columns() -> list[str]:
    cols = [f"pr_{h:03d}" for h in range(1, MAX_LEAD + 1)]
    cols += [f"{v}_{h:02d}" for v in ("t2", "td") for h in T_LEADS]
    return cols


def tasks(run: pd.Timestamp) -> list[tuple[str, str, int]]:
    """(key, url, lead) for every file a run needs."""
    out = [("pr1", file_url(run, h, "pr1"), h) for h in P1_LEADS]
    out += [("pr3", file_url(run, h, "pr3"), h) for h in P3_LEADS]
    out += [(v, file_url(run, h, v), h) for v in ("t2", "td") for h in T_LEADS]
    return out


def merge_ranges(spans: list[tuple[int, int]], gap: int = 64 * 1024) -> list[tuple[int, int]]:
    """Coalesce (offset, size) byte spans that are at most `gap` apart into
    fewer range requests."""
    out: list[list[int]] = []
    for off, size in sorted(spans):
        if out and off - (out[-1][0] + out[-1][1]) <= gap:
            out[-1][1] = max(out[-1][1], off + size - out[-1][0])
        else:
            out.append([off, size])
    return [(a, b) for a, b in out]


def stations() -> pd.DataFrame:
    """Every station of every league, once (station_id index, latitude,
    longitude, elevation_ft)."""
    from .contract import stations as league_stations

    frames = []
    for lg in LEAGUES:
        try:
            frames.append(league_stations(lg)[["latitude", "longitude", "elevation_ft"]])
        except Exception as e:  # a league missing from this PowderBench checkout
            print(f"ukmo: no stations for {lg}: {type(e).__name__}", flush=True)
    st = pd.concat(frames)
    return st[~st.index.duplicated()].sort_index()


# ---- reading one file (runs inside a worker process) ----

_S = None


def _session():
    global _S
    if _S is None:
        import requests

        _S = requests.Session()
    return _S


@cache
def _grid(run: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """The model grid (the same for every run): latitudes, longitudes."""
    import fsspec
    import h5py

    url = file_url(run, 0, "td")
    with fsspec.filesystem("http").open(url, "rb", block_size=32 * 1024, cache_type="blockcache") as fo:
        f = h5py.File(fo)
        return f["latitude"][:].astype(np.float64), f["longitude"][:].astype(np.float64)


def grid_index(lat: np.ndarray, lon: np.ndarray, run: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Nearest model grid cell (row, column) of each point."""
    glat, glon = _grid(run)
    iy = np.abs(glat[None, :] - np.asarray(lat)[:, None]).argmin(1)
    ix = np.abs(((glon[None, :] - np.asarray(lon)[:, None]) + 180) % 360 - 180).argmin(1)
    return iy, ix


def read_points(url: str, iy: np.ndarray, ix: np.ndarray, levels: tuple[int, ...] = ()) -> np.ndarray | None:
    """Values at grid cells (iy, ix) of the file's main field; with `levels`
    (hPa), one row per level. None when the file does not exist."""
    import fsspec
    import h5py

    try:
        fo = fsspec.filesystem("http").open(url, "rb", block_size=32 * 1024, cache_type="blockcache")
    except FileNotFoundError:
        return None
    with fo:
        f = h5py.File(fo)
        # the field itself, not the coordinate bounds or the pressure-level files' `flag` mask
        d = next(v for k, v in f.items() if isinstance(v, h5py.Dataset) and v.ndim >= 2 and v.size > 10_000
                 and k != "flag" and not k.endswith("_bnds"))
        lev_idx = [None]
        if d.ndim == 3:
            p = f["pressure"][:]
            lev_idx = [int(np.argmin(np.abs(p - lv * 100))) for lv in levels]
        cy, cx = iy // CHUNK, ix // CHUNK
        blocks = sorted(set(zip(cy.tolist(), cx.tolist())))
        want = []  # (level slot, chunk, chunk info)
        for k, li in enumerate(lev_idx):
            for b in blocks:
                coord = (b[0] * CHUNK, b[1] * CHUNK) if li is None else (li, b[0] * CHUNK, b[1] * CHUNK)
                want.append((k, b, d.id.get_chunk_info_by_coord(coord)))
        dtype = d.dtype
        fill = d.attrs.get("_FillValue")
        scale, offset = d.attrs.get("scale_factor"), d.attrs.get("add_offset")
    spans = merge_ranges([(ci.byte_offset, ci.size) for _, _, ci in want])
    raw = {}
    s = _session()
    for off, size in spans:
        r = s.get(url, headers={"Range": f"bytes={off}-{off + size - 1}"}, timeout=60)
        if r.status_code not in (200, 206):
            raise OSError(f"{url}: HTTP {r.status_code}")
        raw[off] = r.content
    out = np.full((len(lev_idx), len(iy)), np.nan, np.float32)
    for k, b, ci in want:
        base = next(o for o, _ in spans if o <= ci.byte_offset < o + len(raw[o]))
        buf = raw[base][ci.byte_offset - base: ci.byte_offset - base + ci.size]
        a = np.frombuffer(zlib.decompress(buf), dtype=dtype).reshape(CHUNK, CHUNK).astype(np.float32)
        m = (cy == b[0]) & (cx == b[1])
        out[k, m] = a[iy[m] - b[0] * CHUNK, ix[m] - b[1] * CHUNK]
    if fill is not None:
        out[out == np.float32(fill)] = np.nan
    if scale is not None:
        out = out * np.float32(scale) + np.float32(offset or 0)
    return out if levels else out[0]


def _read_task(args):
    global _S
    key, url, lead, iy, ix = args
    for attempt in range(5):  # S3 drops a connection now and then
        try:
            return key, lead, read_points(url, iy, ix, LEVELS if key in ("t", "z") else ())
        except Exception:
            if attempt == 4:
                raise
            _S = None
            time.sleep(3 * (attempt + 1))


# ---- one run, one unit ----

def run_exists(run: pd.Timestamp) -> bool:
    import requests

    return requests.head(file_url(run, 0, "td"), timeout=30).status_code == 200


def harvest_run(run: pd.Timestamp, st: pd.DataFrame, iy, ix, pool) -> pd.DataFrame | None:
    """One row per station: the run's hourly precipitation and 3-hourly
    temperatures; None when the run is not in the bucket."""
    if not run_exists(run):
        return None
    got = {}
    for key, lead, v in pool.map(_read_task, [(k, u, h, iy, ix) for k, u, h in tasks(run)], chunksize=2):
        got[key, lead] = v
    n = len(st)
    nan = np.full(n, np.nan, np.float32)
    cols = {}
    for h in P1_LEADS:
        v = got.get(("pr1", h))
        cols[f"pr_{h:03d}"] = nan if v is None else v * 1000  # m -> mm
    for h in P3_LEADS:
        v = got.get(("pr3", h))
        for j in (h - 2, h - 1, h):
            cols[f"pr_{j:03d}"] = nan if v is None else v * 1000 / 3
    for v in ("t2", "td"):
        for h in T_LEADS:
            a = got.get((v, h))
            cols[f"{v}_{h:02d}"] = nan if a is None else a - 273.15
    df = pd.DataFrame({c: np.round(cols[c], 2).astype(np.float32) for c in columns()})
    df.insert(0, "run", run)
    df.insert(0, "station_id", st.index.values)
    return df


def runs_in(hour: int, month: pd.Period, now: pd.Timestamp) -> list[pd.Timestamp]:
    days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
    runs = [d + pd.Timedelta(hours=hour) for d in days]
    return [r for r in runs if r >= FIRST and r <= now - pd.Timedelta(hours=8)]


def units(now: pd.Timestamp) -> list[tuple[int, pd.Period]]:
    """Every complete month (the current one waits until it is over), oldest
    first: the bucket deletes from the old end."""
    last = (now - pd.Timedelta(days=1)).to_period("M") - 1
    months = pd.period_range(FIRST.to_period("M"), last, freq="M")
    return [(h, m) for h in RUN_HOURS for m in months]


def run(minutes: float, workers: int = 16, hours: tuple[int, ...] = RUN_HOURS) -> dict:
    now = pd.Timestamp.utcnow().tz_localize(None)
    deadline = time.time() + minutes * 60
    have = store.remote_assets(TAG) | {p.name for p in DIR.glob(f"{TAG}__*.parquet")}
    todo = [u for u in units(now) if u[0] in hours and unit_name(*u) not in have]
    # oldest month first across both run hours: that end of the bucket disappears first
    todo.sort(key=lambda u: (u[1], RUN_HOURS.index(u[0])))
    print(f"ukmo: {len(todo)} units to go, {minutes:.0f} min budget, {workers} workers", flush=True)
    if not todo:
        return {"done": 0, "remaining": 0}
    DIR.mkdir(parents=True, exist_ok=True)
    st = stations()
    iy, ix = grid_index(st["latitude"].values, st["longitude"].values,
                        (now - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=12))
    print(f"ukmo: {len(st)} stations in {len(set(zip(iy // CHUNK, ix // CHUNK)))} chunks", flush=True)
    done = 0
    with ProcessPoolExecutor(workers) as pool:
        for hour, month in todo:
            t0 = time.time()
            frames, missing = [], 0
            for r in runs_in(hour, month, now):
                if time.time() > deadline:
                    print("  time budget reached", flush=True)
                    return {"done": done, "remaining": len(todo) - done}
                try:
                    df = harvest_run(r, st, iy, ix, pool)
                except Exception as e:  # keep the unit; a run that fails twice stays missing
                    print(f"  {r}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                    df = None
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


def load(hour: int = 12) -> pd.DataFrame:
    files = sorted(DIR.glob(f"{TAG}__{hour:02d}z__*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


# ---- features: the same daily rows and round windows as an Open-Meteo model ----

MODEL = "ukmo_mo"
LAPSE_K_PER_M = 0.0065


def hourly(raw: pd.DataFrame, st: pd.DataFrame, delta_m: pd.Series) -> pd.DataFrame:
    """Harvested runs (one run hour) -> `features.daily` input: hourly
    precipitation and 2 m temperature moved to the station's height, in
    station-local time. `lead` = local days after the run's UTC date, so
    day D from the 12Z run of D-1 is lead 1, as in Open-Meteo's
    previous-runs archive."""
    from .ens_features import station_tz

    h = np.arange(1, MAX_LEAD + 1)
    P = raw[[f"pr_{x:03d}" for x in h]].to_numpy(np.float32)
    T3 = raw[[f"t2_{x:02d}" for x in T_LEADS]].to_numpy(np.float32)
    lo = np.minimum(h // 3, len(T_LEADS) - 2)
    fr = ((h - lo * 3) / 3.0).astype(np.float32)
    T = T3[:, lo] * (1 - fr) + T3[:, lo + 1] * fr  # hourly, linear between the 3-hourly values
    dz = delta_m.reindex(raw["station_id"]).fillna(0.0).to_numpy(np.float32)
    T -= LAPSE_K_PER_M * dz[:, None]
    tz = station_tz(st.loc[st.index.isin(raw["station_id"].unique())])
    n = len(raw)
    sid = np.repeat(raw["station_id"].to_numpy(), len(h))
    run = np.repeat(pd.to_datetime(raw["run"]).to_numpy(), len(h))
    utc = pd.DatetimeIndex(run + np.tile(h.astype("timedelta64[h]"), n))
    local = np.empty(len(utc), dtype="datetime64[ns]")
    zones = tz.reindex(sid).to_numpy()
    for z in pd.unique(zones):
        m = zones == z
        local[m] = utc[m].tz_localize("UTC").tz_convert(z).tz_localize(None).to_numpy()
    local = pd.DatetimeIndex(local)
    day = (local - pd.Timedelta(hours=1)).normalize()
    lead = ((day - pd.DatetimeIndex(run).normalize()) // pd.Timedelta(days=1)).astype(np.int16)
    return pd.DataFrame({"station_id": sid, "model": MODEL, "lead": lead, "time": local,
                         "precipitation": P.ravel(), "temperature_2m": T.ravel(), "snowfall": np.nan})


def windows(hour: int = 12) -> pd.DataFrame:
    """Per (station_id, round_date, horizon_h): `<var>__ukmo_mo` columns built
    like the Open-Meteo models' (features.daily -> archive_windows)."""
    from . import features

    files = sorted(DIR.glob(f"{TAG}__{hour:02d}z__*.parquet"))
    if not files:
        return pd.DataFrame(columns=["station_id", "round_date", "horizon_h"])
    st = stations()
    terr = features.terrain()
    delta = terr["elev_delta_14km"] if "elev_delta_14km" in terr else pd.Series(dtype=float)
    dly = pd.concat([features.daily(hourly(pd.read_parquet(f), st, delta)) for f in files], ignore_index=True)
    win = features.archive_windows(dly)
    keys = ["station_id", "round_date", "horizon_h"]
    p = win.pivot_table(index=keys, columns="model", values=list(features.WIN_VARS), aggfunc="first")
    p.columns = [f"{v}__{m}" for v, m in p.columns]
    return p.reset_index()
