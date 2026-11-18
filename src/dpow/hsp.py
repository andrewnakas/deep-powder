"""Hourly snowpack right up to the cutoff (stations league, NRCS AWDB).

The daily snowpack features (snowpack.py) can only use the end-of-day reading
of D-2: for a round whose cutoff is 00Z on D, a storm that fell overnight
into D-1 is invisible to them. SNOTEL also reports hourly, about two hours
behind real time, so the snow that fell in the last day and a half before
the cutoff is measurable. It matters for the score in two ways: fresh snow
settles fast (the depth change the truth is built from comes in low for
days after a storm), and a storm in progress at the cutoff is often still
going when the window opens.

Readings used: the newest hourly values at least LAG_H hours before the
league's cutoff (training and live alike). Noisy single hours are smoothed
by taking the median of three readings at each point.

Columns (inches; depth = SNWD, water = WTEQ), per station and round (same on every horizon):
  hsp_depth                 snow depth now
  hsp_chg24, hsp_chg48      depth change over the last 24 / 48 h (negative = settling or melt)
  hsp_swe_chg24, hsp_swe_chg48   water gained over the last 24 / 48 h
  _hsp_age                  hours from the newest reading used to the cutoff (not a feature:
                            live it runs a few hours older than in training)
"""

from __future__ import annotations

import time
from datetime import date
from functools import cache

import numpy as np
import pandas as pd

from . import HORIZONS

NAME = "awdb-hourly"
PREFIX = "hsp"
TAG = "nbhd"
FILES = "hsp"
LEAGUES = ("stations",)
AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
AWDB_SUFFIXES = (":SNTL", ":SNTLT", ":MSNT")
LAG_H = 3.0  # newest reading at least this long before the cutoff (AWDB posts ~2 h behind)
SPAN_H = 54  # hours of readings fetched before that point
BATCH = 40  # stations per request (AWDB throttles bursts from CI runners: keep requests few and slow)
COLS = ("hsp_depth", "hsp_chg24", "hsp_chg48", "hsp_swe_chg24", "hsp_swe_chg48", "_hsp_age")  # age: diagnostic only


def _get(path: str, params: dict, retries: int = 6):
    import requests

    for attempt in range(retries):
        try:
            r = requests.get(f"{AWDB}/{path}", params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"AWDB {path} kept failing")


@cache
def _tz(triplets: tuple[str, ...]) -> dict[str, float]:
    """Station -> UTC offset (h) of its hourly timestamps (AWDB dataTimeZone; SNOTEL uses -8)."""
    out = {}
    for i in range(0, len(triplets), 100):
        for s in _get("stations", {"stationTriplets": ",".join(triplets[i:i + 100])}) or []:
            out[s["stationTriplet"]] = float(s.get("dataTimeZone") or -8.0)
    return out


def _cutoff(league: str, round_date: date) -> pd.Timestamp:
    from . import use_pb_data

    use_pb_data()
    from powderbench.leagues import get_league

    return pd.Timestamp(get_league(league).cutoff_utc(round_date)).tz_localize(None)


def _hourly(triplets: list[str], start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """-> station_id, utc, SNWD, WTEQ (hourly, inches)."""
    tz = _tz(tuple(sorted(triplets)))
    rows = []
    for i in range(0, len(triplets), BATCH):
        batch = triplets[i:i + BATCH]
        # AWDB windows are in station time; -8 h covers every SNOTEL station, the margin covers the rest
        b = (start_utc - pd.Timedelta(hours=10)).strftime("%Y-%m-%d %H:00")
        e = (end_utc - pd.Timedelta(hours=4)).strftime("%Y-%m-%d %H:00")
        data = _get("data", {"stationTriplets": ",".join(batch), "elements": "SNWD,WTEQ",
                             "duration": "HOURLY", "beginDate": b, "endDate": e}) or []
        for s in data:
            sid = s["stationTriplet"]
            off = tz.get(sid, -8.0)
            for el in s.get("data", []):
                code = el["stationElement"]["elementCode"]
                for v in el.get("values", []):
                    if v.get("value") is None:
                        continue
                    t = pd.Timestamp(v["date"]) - pd.Timedelta(hours=off)
                    rows.append((sid, t, code, float(v["value"])))
        time.sleep(0.5)
    if not rows:
        return pd.DataFrame(columns=["station_id", "utc", "SNWD", "WTEQ"])
    d = pd.DataFrame(rows, columns=["station_id", "utc", "el", "v"])
    d = d[(d["utc"] >= start_utc) & (d["utc"] <= end_utc)]
    return d.pivot_table(index=["station_id", "utc"], columns="el", values="v", aggfunc="first").reset_index()


def _at(g: pd.DataFrame, col: str, t: pd.Timestamp, tol_h: float = 3.0) -> float:
    """Median of the (up to) three readings closest to t, within tol_h."""
    if col not in g:
        return np.nan
    x = g[["utc", col]].dropna()
    if x.empty:
        return np.nan
    dt = (x["utc"] - t).abs() / pd.Timedelta(hours=1)
    near = x[dt <= tol_h].assign(dt=dt[dt <= tol_h]).nsmallest(3, "dt")
    return float(near[col].median()) if len(near) else np.nan


def features(st: pd.DataFrame, cutoff: pd.Timestamp, round_date: date) -> pd.DataFrame:
    ids = [s for s in st.index if str(s).endswith(AWDB_SUFFIXES)]
    if not ids:
        return pd.DataFrame()
    last_ok = cutoff - pd.Timedelta(hours=LAG_H)
    h = _hourly(ids, last_ok - pd.Timedelta(hours=SPAN_H), last_ok)
    rows = []
    for sid, g in h.groupby("station_id"):
        have = g.dropna(subset=[c for c in ("SNWD", "WTEQ") if c in g], how="all")
        if have.empty:
            continue
        now = have["utc"].max()
        dep, swe = _at(g, "SNWD", now), _at(g, "WTEQ", now)
        rows.append({
            "station_id": sid, "hsp_depth": dep,
            "hsp_chg24": dep - _at(g, "SNWD", now - pd.Timedelta(hours=24)),
            "hsp_chg48": dep - _at(g, "SNWD", now - pd.Timedelta(hours=48)),
            "hsp_swe_chg24": swe - _at(g, "WTEQ", now - pd.Timedelta(hours=24)),
            "hsp_swe_chg48": swe - _at(g, "WTEQ", now - pd.Timedelta(hours=48)),
            "_hsp_age": (cutoff - now) / pd.Timedelta(hours=1),
        })
    if not rows:
        return pd.DataFrame()
    f = pd.DataFrame(rows)
    f[list(COLS)] = f[list(COLS)].astype("float32")
    out = pd.concat([f.assign(horizon_h=hh) for hh in HORIZONS], ignore_index=True)
    out["round_date"] = round_date
    return out


def for_round(league: str, round_date: date, st: pd.DataFrame | None = None, latest: bool = False) -> pd.DataFrame:
    """Same rule in training and live (`latest` is accepted for the harvest interface; readings
    are always the newest LAG_H hours before the cutoff)."""
    from . import contract

    if league not in LEAGUES:
        return pd.DataFrame()
    st = contract.stations(league) if st is None else st
    return features(st, _cutoff(league, round_date), round_date)


def asset(league: str, season: int) -> str:
    return f"{FILES}__{league}__{int(season)}.parquet"


def join(df: pd.DataFrame) -> pd.DataFrame:
    from .nbhd import join_files

    return join_files(df, FILES, PREFIX, "hourly snowpack", key="depth")
