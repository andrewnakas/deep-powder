"""Open-Meteo-style model columns for every station from dynamical.org's
deterministic archives: GFS (from 2021-05) and HRRR (from 2018-08, CONUS).

Open-Meteo's `gfs_seamless` and `ncep_hrrr_conus` columns are on only the
featured stations and a sample (11-26% of table rows). The same runs are in
dynamical.org's Icechunk stores for every grid point, so the 12Z run of each
day is read at every station and laid out like the Met Office harvest
(ukmo.py: `pr_001`..`pr_096` hourly precipitation in mm, `t2_00`..`t2_96`
2 m temperature in C every 3 h), and `ukmo.windows_from` builds the same
round windows as the Open-Meteo models'. Units harvested from 2026-11-23 on
also carry `sn_001`..`sn_096`, hourly snowfall in cm derived like Open-Meteo's
NCEP snowfall: precipitation x percent frozen / 100 x 0.7 cm per mm.

A unit = (dataset, month): `dynom__<ds>__12z__<YYYY-MM>.parquet` assets of
the `dynom` release. Resumable; `part`/`parts` split the units between jobs.
`update` keeps this month's and last month's units current every day, and
`dynom__<ds>__windows__12z.parquet` holds the round windows of every unit.

Adopted 2026-11-23 (`gfs_fill`: +0.46 pooled, every season, league and
horizon up): `fill` puts GFS into Open-Meteo's `gfs_seamless` columns
wherever Open-Meteo gave none, in the training table (cli `_build`) and live
(`predict.build`, from `round_windows`). HRRR (`dyn_fill`, +0.44) added
nothing on top, so it is harvested but not filled (`FILL`).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import WORK, store
from .ukmo import MAX_LEAD, SNOW_COLS, T_LEADS, columns, hourly, lapse_delta, pivot, stations, windows_from

TAG = "dynom"
DIR = WORK / "dynom"
RUN_HOUR = 12
MODELS = {"gfs": "gfs_dyn", "hrrr": "hrrr_dyn"}  # dataset -> model name in the columns
OM_TWIN = {"gfs": "gfs_seamless", "hrrr": "ncep_hrrr_conus"}  # the Open-Meteo model it stands in for
# filled into the table and live: GFS alone (gfs_fill +0.46; adding HRRR, dyn_fill, +0.44, bought nothing)
FILL = ("gfs",)
CM_PER_MM = 0.7  # Open-Meteo's snowfall from water-equivalent precipitation
KEYS = ["station_id", "round_date", "horizon_h"]


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
    # GFS/HRRR write percent frozen as -50 where nothing falls
    fz = np.clip(d["pfz"].to_numpy(np.float32)[ok], 0, 100) / 100 if "pfz" in d else np.full(len(p), np.nan, np.float32)
    s = p * fz * CM_PER_MM
    P = np.full((n, MAX_LEAD), np.nan, np.float32)
    S = np.full((n, MAX_LEAD), np.nan, np.float32)
    for k in range(int(dt.max()) if len(dt) else 0):  # hour k before the step's end
        h = lead - k
        m = (k < dt) & (h >= 1) & (h <= MAX_LEAD)
        P[i[m], h[m] - 1] = p[m] / dt[m]
        S[i[m], h[m] - 1] = np.where(p[m] > 0, s[m], 0.0) / dt[m]
    T = np.full((n, len(T_LEADS)), np.nan, np.float32)
    tl = {h: j for j, h in enumerate(T_LEADS)}
    j = np.array([tl.get(x, -1) for x in lead])
    m = j >= 0
    T[i[m], j[m]] = t[m]
    cols = {f"pr_{h:03d}": P[:, h - 1] for h in range(1, MAX_LEAD + 1)}
    cols |= {f"t2_{h:02d}": T[:, j] for j, h in enumerate(T_LEADS)}
    cols |= {f"td_{h:02d}": np.full(n, np.nan, np.float32) for h in T_LEADS}
    cols |= {c: S[:, h] for h, c in enumerate(SNOW_COLS)}
    df = pd.DataFrame({c: np.round(cols[c], 2).astype(np.float32) for c in (*columns(), *SNOW_COLS)})
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


def windows_name(ds: str) -> str:
    return f"{TAG}__{ds}__windows__{RUN_HOUR:02d}z.parquet"


def _typed(w: pd.DataFrame) -> pd.DataFrame:
    w["round_date"] = pd.to_datetime(w["round_date"]).dt.date
    return w.astype({c: "float32" for c in w.columns if "__" in c})


def load_windows(ds: str) -> pd.DataFrame:
    """Every unit's round windows: the release's windows file, else built from
    the units (and published, so the next job only downloads it)."""
    p = DIR / windows_name(ds)
    store.pull(TAG, DIR, p.name, overwrite=True)
    if not p.exists():
        pull(ds)
        w = windows(ds)
        if w.empty:
            return w
        DIR.mkdir(parents=True, exist_ok=True)
        w.to_parquet(p, index=False, compression="zstd")
        try:
            store.push(TAG, [p])
        except RuntimeError as e:  # a read-only job (experiment.yml) builds it for itself
            print(f"  {p.name}: not published ({str(e)[:80]})", flush=True)
    return _typed(pd.read_parquet(p))


def _lag(ds: str) -> pd.Timedelta:
    from .sources.dynamical import DATASETS

    return pd.Timedelta(hours=DATASETS[ds].latency_h + 1)


def update(datasets: list[str], minutes: float) -> dict:
    """Daily: the 12Z runs of this month and last month that are out and not
    yet in their month's unit are added to it (a unit can be partial until
    its month ends); then the windows file's rounds from last month on are
    rebuilt from the updated units."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    deadline = time.time() + minutes * 60
    have = store.remote_assets(TAG)
    st = stations()
    DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for ds in datasets:
        months = [now.to_period("M") - 1, now.to_period("M")]
        added = 0
        for m in months:
            p = DIR / unit_name(ds, m)
            if p.name in have:
                store.pull(TAG, DIR, p.name, overwrite=True)
            old = pd.read_parquet(p) if p.exists() else None
            done = set(pd.to_datetime(old["run"])) if old is not None else set()
            days = pd.date_range(m.start_time, m.end_time.normalize(), freq="D")
            runs = [r for r in (d + pd.Timedelta(hours=RUN_HOUR) for d in days)
                    if r <= now - _lag(ds) and r not in done]
            frames = []
            for r in runs:
                if time.time() > deadline:
                    break
                try:
                    df = harvest_run(ds, r, st)
                except Exception as e:  # stays missing; tried again tomorrow
                    print(f"  {ds} {r}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                    df = None
                if df is not None:
                    frames.append(df)
            if frames:
                both = [f for f in (old, *frames) if f is not None]
                # older units have no snowfall columns: concat leaves them NaN there
                pd.concat(both, ignore_index=True).sort_values(["run", "station_id"]).to_parquet(
                    p, index=False, compression="zstd")
                store.push(TAG, [p])
                added += len(frames)
            print(f"  {p.name}: +{len(frames)} runs ({len(runs) - len(frames)} not out or failed)", flush=True)
        if added or windows_name(ds) not in have:
            _refresh_windows(ds, months[0], have)
        out[ds] = added
    return out


def _refresh_windows(ds: str, first: pd.Period, have: set[str]) -> None:
    """The windows file with its rounds from `first` on rebuilt from the units
    (the month before `first` supplies the day before its first round)."""
    p = DIR / windows_name(ds)
    if p.name not in have:
        load_windows(ds)  # builds the whole file from every unit
        return
    store.pull(TAG, DIR, p.name, overwrite=True)
    prev = unit_name(ds, first - 1)
    if prev in have:
        store.pull(TAG, DIR, prev, overwrite=False)
    files = [DIR / unit_name(ds, m) for m in (first - 1, first, first + 1)]
    tail = _typed(windows_from([f for f in files if f.exists()], MODELS[ds]))
    start = first.start_time.date()
    old = _typed(pd.read_parquet(p))
    w = pd.concat([old[old["round_date"] < start], tail[tail["round_date"] >= start]], ignore_index=True)
    w.sort_values(KEYS).to_parquet(p, index=False, compression="zstd")
    store.push(TAG, [p])
    print(f"  {p.name}: {len(w)} rows, rounds from {start} rebuilt ({len(tail)} windows)", flush=True)


def round_windows(ds: str, round_date, league: str, st: pd.DataFrame) -> pd.DataFrame:
    """Live: one round's `<var>__<model>` windows from the run the training
    rows use (the 12Z run of the day before the round), or, when the league's
    cutoff comes before it is out, the newest 12Z run that is. Its lead days
    are shifted so day i of the round is read like the training rows' lead
    i+1."""
    from . import features
    from .ens_features import init_for_round

    rd = pd.Timestamp(round_date)
    want = init_for_round(round_date, league, ds)
    newest = want.normalize() + pd.Timedelta(hours=RUN_HOUR)
    if newest > want:
        newest -= pd.Timedelta(days=1)
    run = min(rd - pd.Timedelta(days=1) + pd.Timedelta(hours=RUN_HOUR), newest)
    raw = None
    for back in range(3):  # a run missing from the store: the one before it
        r = run - pd.Timedelta(days=back)
        raw = harvest_run(ds, r, st)
        if raw is not None:
            run = r
            break
    if raw is None:
        return pd.DataFrame(columns=KEYS)
    dly = features.daily(hourly(raw, st, lapse_delta(), MODELS[ds]))
    dly["lead"] = dly["lead"] - (rd - pd.Timedelta(days=1) - run.normalize()).days
    win = features.archive_windows(dly)
    win = win[pd.to_datetime(win["round_date"]) == rd]
    return _typed(pivot(win)) if len(win) else pd.DataFrame(columns=KEYS)


def fill(df: pd.DataFrame, wins: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Open-Meteo's GFS/HRRR columns, where Open-Meteo gave none, from the same
    model read from dynamical.org (`wins`: dataset -> its round windows); the
    cross-model summaries and ref_snow rebuilt on the rows that changed.
    Twin columns missing from `df` (a live round with no Open-Meteo block)
    are created."""
    from . import features
    from .dataset import add_ref

    d = df.copy(deep=False)
    changed = np.zeros(len(d), bool)
    for ds, w in wins.items():
        new = [c for c in w.columns if "__" in c] if w is not None else []
        if not new:
            continue
        k = d[KEYS].merge(w, on=KEYS, how="left")
        for c in new:
            twin = c.rsplit("__", 1)[0] + "__" + OM_TWIN[ds]
            v = k[c].to_numpy(np.float32)
            cur = d[twin].to_numpy(np.float32) if twin in d else np.full(len(d), np.nan, np.float32)
            take = np.isnan(cur) & ~np.isnan(v)
            if take.any():
                d[twin] = np.where(take, v, cur)
                changed |= take
    if changed.any():
        cols = [c for c in d.columns if c.startswith(features.PER_MODEL)]
        sub = d.loc[changed, cols + [c for c in ("gefs_snow_q50", "ecens_snow_q50") if c in d]].copy()
        features.summarize(sub)
        add_ref(sub)
        for c in (*features.SUMMARY_COLS, "ref_snow"):
            if c not in d:
                d[c] = np.float32(np.nan)
            d.loc[changed, c] = sub[c].to_numpy()
    print(f"  dynom fill: {changed.mean():.0%} of rows", flush=True)
    return d
