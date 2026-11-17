"""Long-record station calibration: each station against what actually fell.

Forecast archives start in 2018-2024, so a per-station factor learned from
forecasts rests on a few seasons. Analyses go back much further: HRRR's
hourly 3 km analysis to 2014-10 (CONUS) and GEFS's 3-hourly 0.25 degree
analysis to 2000 (global). With PowderBench's own truth (stations from 2015,
observers 2016, alps 2000) that gives every station up to a dozen seasons of
"reported snow per inch of snow that fell at its grid cell", a stable measure
of where the station sits relative to the grid (ridge, valley, lee, the
observer's habits).

`harvest` writes daily analysis snow per station (release `nbhd`, files
ana__<league>__<source>.parquet); `ratios` turns it into per-station,
per-season-history ratios; `add_history` adds them to the training table as
`ana_ratio_<source>` (+ the analysis snow per valid day as evidence count).
History only: a row of season s uses seasons before s.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .ens_features import MM_PER_IN, slr_2m, station_tz

TAG = "nbhd"
SOURCES = {
    # name: (bucket, prefix, first date, hours per step, has native snowfall)
    "hrrr": ("dynamical-noaa-hrrr", "noaa-hrrr-analysis/v0.2.0.icechunk", "2014-10-01", 1, True),
    "gefs": ("dynamical-noaa-gefs", "noaa-gefs-analysis/v0.1.2.icechunk", "2000-01-01", 3, False),
}
LEAGUES = {"hrrr": ("stations", "observers"), "gefs": ("stations", "observers", "alps")}
TRUTH_FROM = {"stations": "2014-10-01", "observers": "2015-10-01", "alps": "2000-10-01"}  # first winter with truth
K = 10.0  # inches of pseudo-evidence toward a ratio of 1
CLIP = (0.2, 5.0)
MIN_DAYS = 60  # valid truth days before a station gets a ratio


def _open(source: str):
    import icechunk as ic
    import xarray as xr

    bucket, prefix, *_ = SOURCES[source]
    st = ic.s3_storage(bucket=bucket, prefix=prefix, region="us-west-2", anonymous=True)
    return xr.open_zarr(ic.Repository.open(st).readonly_session("main").store, consolidated=False, chunks=None)


def _cells(ds, st: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = st["latitude"].to_numpy(), ((st["longitude"].to_numpy() + 180) % 360) - 180
    if "x" in ds.dims:
        from scipy.spatial import cKDTree

        la, lo = np.radians(ds["latitude"].values), np.radians(ds["longitude"].values)
        xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1).reshape(-1, 3)
        rl, ro = np.radians(lat), np.radians(lon)
        q = np.stack([np.cos(rl) * np.cos(ro), np.cos(rl) * np.sin(ro), np.sin(rl)], -1)
        _, idx = cKDTree(xyz).query(q)
        return np.unravel_index(idx, ds["latitude"].shape)
    yi = np.abs(ds["latitude"].values[:, None] - lat[None]).argmin(0)
    xi = np.abs(ds["longitude"].values[:, None] - lon[None]).argmin(0)
    return yi, xi


def _season_ranges(first: str, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Oct 1 - May 31 of every winter from `first` to `end`."""
    out = []
    for y in range(pd.Timestamp(first).year - 1, end.year + 1):
        a, b = max(pd.Timestamp(f"{y}-10-01"), pd.Timestamp(first)), min(pd.Timestamp(f"{y + 1}-06-01"), end)
        if a < b:
            out.append((a, b))
    return out


def harvest(league: str, source: str, st: pd.DataFrame, start: str | None = None, minutes: float = 300,
            done: set | None = None) -> tuple[pd.DataFrame, set]:
    """Daily (station-local) analysis snow, inches: `snow` (precipitation type x
    2 m temperature ratio) and, for HRRR, `nat` (its own snowfall field).
    Works one winter at a time; `done` = winters already harvested."""
    import xarray as xr

    ds = _open(source)
    _, _, first, step_h, has_nat = SOURCES[source]
    if source == "hrrr":
        from .sources.dynamical import in_conus

        st = st[in_conus(st["latitude"], st["longitude"])]
    yi, xi = _cells(ds, st)
    ydim, xdim = ("y", "x") if "x" in ds.dims else ("latitude", "longitude")
    tz = station_tz(st)
    vars_ = ["precipitation_surface", "percent_frozen_precipitation_surface", "temperature_2m"]
    if has_nat:
        vars_.append("snowfall_surface")
    end = pd.Timestamp(ds["time"].values[-1])
    done = set(done or ())
    t0, parts = time.time(), []
    for a, b in _season_ranges(max(pd.Timestamp(start or first), pd.Timestamp(first)).isoformat(), end):
        key = f"{a.year}"
        if key in done:
            continue
        if time.time() - t0 > minutes * 60:
            break
        for attempt in range(4):
            try:
                v = (ds[vars_].sel(time=slice(a, b - pd.Timedelta(seconds=1)))
                     .isel({ydim: xr.DataArray(yi, dims="pt"), xdim: xr.DataArray(xi, dims="pt")}).load())
                break
            except Exception as e:  # flaky range reads
                if attempt == 3:
                    raise
                print(f"  retry {a.date()}: {type(e).__name__}", flush=True)
                time.sleep(20 * (attempt + 1))
        sec = step_h * 3600.0
        p = np.nan_to_num(v["precipitation_surface"].transpose("pt", "time").to_numpy()).clip(min=0) * sec  # mm
        t = v["temperature_2m"].transpose("pt", "time").to_numpy()
        fz = v["percent_frozen_precipitation_surface"].transpose("pt", "time").to_numpy() / 100.0
        fz = np.where(np.isfinite(fz) & (fz >= 0), fz, 1.0 / (1.0 + np.exp((np.nan_to_num(t) - 1.0) / 0.5)))
        snow = p * np.clip(fz, 0, 1) * slr_2m(np.nan_to_num(t)) / MM_PER_IN
        cols = {"snow": snow}
        if has_nat:
            nat = np.nan_to_num(v["snowfall_surface"].transpose("pt", "time").to_numpy()).clip(min=0)
            cols["nat"] = nat * sec * 1000 / MM_PER_IN
        mid = pd.DatetimeIndex(v["time"].values) - pd.Timedelta(hours=step_h / 2)  # step ends at its time stamp
        for zone, idx in tz.groupby(tz).groups.items():
            rows = st.index.get_indexer(idx)
            day = mid.tz_localize("UTC").tz_convert(zone).tz_localize(None).normalize()
            frames = []
            for k, c in cols.items():
                d = pd.DataFrame(c[rows].T, index=day, columns=list(idx)).groupby(level=0).sum()
                frames.append(d.stack().rename(k))
            d = pd.concat(frames, axis=1).rename_axis(["date", "station_id"]).reset_index()
            parts.append(d)
        done.add(key)
        print(f"  {source} {league} winter {a.year}: {len(st)} stations, {(time.time() - t0) / 60:.0f} min", flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out):
        out["date"] = pd.to_datetime(out["date"]).dt.date
        num = [c for c in ("snow", "nat") if c in out]
        out[num] = out[num].astype("float32")
    return out, done


def asset(league: str, source: str) -> str:
    return f"ana__{league}__{source}.parquet"


def ratios(daily_ana: pd.DataFrame, truth: pd.DataFrame, league: str, last_season: int | None = None) -> pd.DataFrame:
    """Per station and season s, the ratio from every winter before s (seasons
    without truth add nothing; `last_season` extends the grid to the live one).
    -> station_id, season, ana_ratio_<col>, ana_days"""
    from .dataset import season_of

    m = truth[truth["valid"].astype(bool)].merge(daily_ana, on=["station_id", "date"])
    m["season"] = season_of(pd.Series(pd.to_datetime(m["date"])), league).to_numpy()
    cols = [c for c in ("snow", "nat") if c in m]
    aggs = {"t": ("snow24", "sum"), "n": ("snow24", "size"), **{c: (c, "sum") for c in cols}}
    g = m.groupby(["station_id", "season"]).agg(**aggs)
    seasons = range(int(m["season"].min()), int(max(m["season"].max(), last_season or 0)) + 2)
    full = pd.MultiIndex.from_product([g.index.levels[0], seasons], names=["station_id", "season"])
    g = g.reindex(full, fill_value=0).groupby(level=0).cumsum().groupby(level=0).shift(1)  # strictly before s
    out = pd.DataFrame(index=g.index)
    ok = g["n"] >= MIN_DAYS
    for c in cols:
        out[f"ana_ratio_{c}"] = ((g["t"] + K) / (g[c] + K)).clip(*CLIP).where(ok).astype("float32")
    out["ana_days"] = g["n"].astype("float32")
    return out.reset_index()


def join(df: pd.DataFrame) -> pd.DataFrame:
    """Add the history ratios (every source and league harvested) to a training table."""
    from . import WORK, store
    from .truth import hf_daily

    dest = WORK / TAG
    store.pull(TAG, dest, "ana__*.parquet")
    df = df.drop(columns=[c for c in df.columns if c.startswith("ana_")])
    out = []
    for lg, part in df.groupby("league", sort=False):
        add = None
        files = sorted(dest.glob(f"ana__{lg}__*.parquet"))
        if files:
            tr = hf_daily(lg)
            for f in files:
                src = f.stem.split("__")[2]
                r = ratios(pd.read_parquet(f), tr, lg, int(part["season"].max()))
                r = r.rename(columns={c: f"{c}_{src}" if c.startswith("ana_ratio") else f"ana_days_{src}"
                                      for c in r.columns if c.startswith("ana_")})
                add = r if add is None else add.merge(r, on=["station_id", "season"], how="outer")
        out.append(part if add is None else part.merge(add, on=["station_id", "season"], how="left"))
    res = pd.concat(out, ignore_index=True)
    rc = [c for c in res.columns if c.startswith("ana_ratio")]
    if rc:
        print(f"  ana: {res[rc].notna().any(axis=1).mean():.0%} of rows have a long-record station ratio", flush=True)
    return res
