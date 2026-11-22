"""The every-station archive: dynamical.org round features for every station
of a league, not only the featured ones and the fixed network sample that
open-powder-ensemble harvests, and ECMWF ENS from the start of dynamical.org's
archive (2020-10) instead of 2024-04. Nothing here calls Open-Meteo.

A unit = (dataset, league, month): all the league's stations on the 2nd,
4th, ... of the month and the 1st of the next one. These are the sibling's
round dates, so the hourly-GFS and
storm-wind features already harvested for those days (nbhd.py, wndx.py, which
cover every station) line up. The features come from the same
`features_for_init` the live forecast uses. Units are
`ens__<ds><tag>__<league>-all__<YYYY-MM>.parquet` assets of this repo's
`arch-<year>` releases. Finished units are skipped, so each job continues
where the last one stopped.

Reading is chunk-bound: the stores' chunks span 4-60 degrees. All ~1,000
stations of a league cost about twice what its 45-100 featured ones do
(ECMWF ENS, one run: 40 s for 45 stations, 80 s for 996).
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd

from . import WORK, store

ARCH = WORK / "arch"
ALL = "-all"
PREFIX = "arch"
NH_MONTHS = (10, 11, 12, 1, 2, 3, 4, 5)
SH_MONTHS = (5, 6, 7, 8, 9, 10)
EVERY_N_DAYS = 2
# open-powder-ensemble's ECMWF ENS units start here (its old store did); every
# earlier month is new for featured stations too
SIBLING_ECENS_FROM = pd.Period("2024-04", freq="M")
KEYS = ["station_id", "round_date", "horizon_h"]


def unit_name(ds: str, league: str, month: pd.Period) -> str:
    from .sources.dynamical import DATASETS

    return f"ens__{ds}{DATASETS[ds].tag}__{league}{ALL}__{month}.parquet"


def release(month: pd.Period) -> str:
    return f"{PREFIX}-{month.year}"


def _all_featured(league: str) -> bool:
    from .contract import stations

    return bool(stations(league)["featured"].all())


def units(leagues: list[str], end: date, datasets: list[str] | None = None) -> list[tuple[str, str, pd.Period]]:
    """Every unit to harvest, most useful first: ECMWF ENS months the sibling
    never had, then the winter core, newest season first."""
    from .sources.dynamical import DATASETS, applies

    out = []
    for lg in leagues:
        only_new_ecens = _all_featured(lg)  # nothing beyond the sibling's units, except older ECMWF ENS
        months = SH_MONTHS if lg in ("era5", "resorts") else NH_MONTHS
        for ds, spec in DATASETS.items():
            if (datasets and ds not in datasets) or not applies(ds, lg):
                continue
            if only_new_ecens and ds != "ecens":
                continue
            for m in pd.period_range(spec.first_init, end, freq="M"):
                if m.month not in months or m.end_time.date() > end:
                    continue
                if only_new_ecens and m >= SIBLING_ECENS_FROM:
                    continue
                out.append((ds, lg, m))
    core = {1: 0, 2: 0, 12: 0, 7: 0, 8: 0, 3: 1, 11: 1, 6: 1, 9: 1, 4: 2, 10: 2, 5: 2}
    new_ecens = lambda u: not (u[0] == "ecens" and u[2] < SIBLING_ECENS_FROM)  # noqa: E731
    return sorted(out, key=lambda u: (new_ecens(u), core.get(u[2].month, 3), -u[2].year, u[0], u[1]))


def harvest_unit(ds: str, league: str, month: pd.Period, deadline: float) -> pd.DataFrame | None:
    """All stations, every 2nd round date of the month (the sibling's dates);
    None when the time budget ran out."""
    from .contract import stations
    from .ens_features import features_for_init, init_for_round

    st = stations(league)
    # the sibling's units hold days 2, 4, ... of the month and the 1st of the next one
    one = pd.Timedelta(days=1)
    rounds = pd.date_range(month.start_time + one, month.end_time.normalize() + one, freq=f"{EVERY_N_DAYS}D").date
    frames = []
    for rd in rounds:
        if time.time() > deadline:
            return None  # unfinished: redone next run (units are atomic)
        try:
            frames.append(features_for_init(ds, init_for_round(rd, league, ds), st, league, round_date=rd))
        except KeyError:  # run missing from the archive
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def releases() -> list[str]:
    r = store._gh("release", "list", "-R", store._repo() or "", "-L", "200", "--json", "tagName", "-q", ".[].tagName")
    return sorted(t for t in r.stdout.split() if t.startswith(PREFIX + "-")) if r.returncode == 0 else []


def remote() -> set[str]:
    return {a for t in releases() for a in store.remote_assets(t)}


def run(leagues: list[str], datasets: list[str] | None, minutes: float, end: date | None = None,
        part: int = 0, parts: int = 1) -> dict:
    """Harvest the next missing units; `part`/`parts` split the list between parallel jobs."""
    end = end or (pd.Timestamp.utcnow() - pd.Timedelta(days=5)).date()
    deadline = time.time() + minutes * 60
    have = remote()
    # split the full list (not just the missing units), so jobs that start at different times agree
    todo = [u for i, u in enumerate(units(leagues, end, datasets)) if i % parts == part and unit_name(*u) not in have]
    print(f"arch: {len(todo)} units to go ({', '.join(leagues)}; {', '.join(datasets or ['all datasets'])}), "
          f"{minutes:.0f} min budget", flush=True)
    ARCH.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for ds, lg, m in todo:
        t0 = time.time()
        try:
            df = harvest_unit(ds, lg, m, deadline)
        except Exception as e:  # one bad unit must not lose the job
            failed += 1
            print(f"  {unit_name(ds, lg, m)}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            if failed > 5:
                break
            continue
        if df is None:
            print("  time budget reached", flush=True)
            break
        p = ARCH / unit_name(ds, lg, m)
        df.to_parquet(p, index=False, compression="zstd")
        store.push(release(m), [p])
        done += 1
        n_st = df["station_id"].nunique() if len(df) else 0
        print(f"  {p.name}: {len(df)} rows, {n_st} stations, {time.time() - t0:.0f}s", flush=True)
    return {"done": done, "failed": failed, "remaining": len(todo) - done}


def pull(leagues: list[str], datasets: list[str] | None = None) -> None:
    from .sources.dynamical import DATASETS

    tags = [f"{ds}{DATASETS[ds].tag}" for ds in (datasets or DATASETS)]
    for t in releases():
        for lg in leagues:
            for tag in tags:
                store.pull(t, ARCH, f"ens__{tag}__{lg}{ALL}__*.parquet", overwrite=False)


def load(league: str, datasets: list[str] | None = None) -> pd.DataFrame:
    """The harvested every-station features of a league, all datasets joined
    on the round keys (the same columns as ens_store.load)."""
    from .sources.dynamical import DATASETS

    out = None
    for ds in datasets or DATASETS:
        files = sorted(ARCH.glob(f"ens__{ds}{DATASETS[ds].tag}__{league}{ALL}__*.parquet"))
        if not files:
            continue
        f = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True)
        f["round_date"] = pd.to_datetime(f["round_date"]).dt.date
        f = f.drop_duplicates(KEYS)
        out = f if out is None else out.merge(f, on=KEYS, how="outer")
    return out if out is not None else pd.DataFrame(columns=KEYS)
