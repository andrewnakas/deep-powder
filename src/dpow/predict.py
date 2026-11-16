"""Build one round's submission.

1. Ensemble features (GEFS, ECMWF ENS, AIFS ENS, GFS, AIFS, HRRR) for
   *every* station in the round, read from dynamical.org (no quota), from
   the run a forecast at this league's cutoff can use.
2. The multi-model Open-Meteo block for featured stations, taken from the
   feature file open-powder-ensemble saved for the same round. deep-powder
   makes no Open-Meteo calls of its own; when that file is missing or stale
   the rows simply carry no multi-model block, as most training rows do.
3. The stack: every level-1 learner fitted on all seasons, then the
   level-2 stacker (blend, dry rule, intervals). The reference LightGBM
   bundle is used if no stack is published, and tier 0 (the ensembles' own
   quantiles) fills any gap, so a file always goes out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import HORIZONS, NH_LEAGUES, OWN_BUNDLE_LEAGUES, PROXY_LEAGUE, QCOLS, ROOT
from .contract import Round, stations
from .dataset import KEYS, assemble
from .ens_features import features_for_init, init_for_round
from .models import fallback, gbm, learners, stacker
from .sources import dynamical
from .store import MODELS

HISTORY = ROOT / "history"
COLS = ["station_id", "horizon_h", "snowfall_in", *QCOLS, "prob_6in", "swe_in"]
SIBLING_FEATURES = ("https://raw.githubusercontent.com/andrewnakas/open-powder-ensemble/main/history/"
                    "{league}/{round_date}.features.parquet")
SIBLING_MAX_AGE_H = 30.0


def bundle_names(league: str) -> list[str]:
    """Preference order: the league's own bundle, then its proxy's."""
    names = [league] if league in OWN_BUNDLE_LEAGUES else []
    base = PROXY_LEAGUE.get(league, league)
    return names + (["nh"] if base in NH_LEAGUES else [])


def load_bundle(league: str) -> tuple[str | None, gbm.Bundle | None]:
    for n in bundle_names(league):
        p = MODELS / f"{n}.pkl"
        if p.exists():
            return n, gbm.Bundle.load(p)
    return None, None


@dataclass
class Stack:
    name: str
    finals: list  # learners.Final
    stacker: stacker.Stacker


def load_stack(league: str) -> Stack | None:
    """The published stack for the league's bundle: needs the stacker and at
    least its anchor learner; other learners are used when their file is there."""
    import pickle

    for n in bundle_names(league):
        sp = MODELS / f"{n}__stacker.pkl"
        if not sp.exists():
            continue
        meta = pickle.loads(sp.read_bytes())
        finals = [pickle.loads(f.read_bytes()) for k in meta["learners"]
                  if (f := MODELS / f"{n}__{k}.pkl").exists()]
        if any(stacker.ANCHOR in fm.models for fm in finals):
            return Stack(n, finals, meta["stacker"])
    return None


def have_model(league: str) -> bool:
    return load_stack(league) is not None or load_bundle(league)[0] is not None


@dataclass
class Result:
    df: pd.DataFrame  # the submission
    raw: pd.DataFrame  # per row: raw_in (model), model_in (after online bias), final, tier
    info: dict
    X: pd.DataFrame | None = None  # what the model saw, featured rows (see save_features)


def live_ensembles(r: Round, st: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Both ensembles at every station; the init a cutoff-time forecast can
    use (or the newest earlier one, if publication runs late)."""
    frames = []
    for name in dynamical.DATASETS:
        if not dynamical.applies(name, r.league):
            continue
        want = init_for_round(r.round_date, r.league, name)
        try:
            inits = dynamical.available_inits(name)
            ok = inits[inits <= want]  # the rule's run, or the newest earlier one if it's late
            if not len(ok):
                continue
            init = ok[-1]
            f = features_for_init(name, init, st, r.league, round_date=r.round_date)
            if f.empty:
                continue
            frames.append(f.set_index(KEYS))
            info[f"{name}_init"] = str(init)[:13]
            print(f"  {r.key}: {name} {str(init)[:13]}Z at {len(st)} stations", flush=True)
        except Exception as e:  # one ensemble missing must not sink the round
            info[f"{name}_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  {r.key}: {name} failed: {e}", flush=True)
    if not frames:
        return pd.DataFrame(columns=KEYS)
    return pd.concat(frames, axis=1).reset_index()


def sibling_det(r: Round, info: dict) -> pd.DataFrame:
    """The Open-Meteo multi-model columns open-powder-ensemble computed for
    this round's featured stations (already limited to the models the archive
    has), keyed station_id, horizon_h. Empty when there is no fresh file."""
    import io

    import requests

    url = SIBLING_FEATURES.format(league=r.league, round_date=r.round_date)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            info["det_block"] = "none yet"
            return pd.DataFrame()
        resp.raise_for_status()
        X = pd.read_parquet(io.BytesIO(resp.content))
    except Exception as e:  # the block is a bonus, never a reason to fail the round
        info["det_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return pd.DataFrame()
    built = pd.to_datetime(X["built_utc"].iloc[0], utc=True) if "built_utc" in X and len(X) else None
    age = (pd.Timestamp.now(tz="UTC") - built).total_seconds() / 3600 if built is not None else np.inf
    cols = [c for c in X.columns if learners.det_column(c) and c != "ref_snow"]
    if age > SIBLING_MAX_AGE_H or not cols:
        info["det_block"] = f"stale ({age:.0f} h)"
        return pd.DataFrame()
    info["det_block"] = f"{X['station_id'].nunique()} stations, built {built:%m-%d %H:%MZ}"
    return X[["station_id", "horizon_h", *cols]].drop_duplicates(["station_id", "horizon_h"])


CAP_MULT, CAP_ADD = 2.5, 0.5  # physical plausibility: never far above the snowiest model


def plausibility_cap(pred: pd.DataFrame, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """A statistical model must not invent a storm no weather model sees (a
    resorts model once forecast 10-14" at La Hoya with every model at 0).
    Each row is capped at CAP_MULT x the snowiest input + CAP_ADD inches, and
    P(>=6") is held near zero when even that cap is under 6"."""
    use = [c for c in cols if c in X]
    if not use or pred.empty:
        return pred
    top = X[use].max(axis=1, skipna=True)
    keys = pd.MultiIndex.from_frame(X[["station_id", "horizon_h"]])
    cap = pd.Series((CAP_MULT * top + CAP_ADD).to_numpy(), index=keys)
    cap = cap[~cap.index.duplicated()]
    c = cap.reindex(pd.MultiIndex.from_frame(pred[["station_id", "horizon_h"]])).to_numpy()
    out = pred.copy()
    ok = ~np.isnan(c)
    for q in (*QCOLS, "snowfall_in"):
        if q in out:
            out.loc[ok, q] = np.minimum(out.loc[ok, q].to_numpy(), c[ok])
    if "prob_6in" in out:
        low = ok & (c < 6.0)
        out.loc[low, "prob_6in"] = np.minimum(out.loc[low, "prob_6in"].fillna(0).to_numpy(), 0.02)
    n = int((ok & (pred["snowfall_in"].to_numpy() > np.nan_to_num(c, nan=np.inf))).sum())
    if n:
        print(f"  plausibility cap trimmed {n} rows", flush=True)
    return out


def _with_density(pred: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """New-snow density (water / depth) from the forecast physics: each
    ensemble's frozen water over its snow depth (members' wet-bulb / 850 hPa
    or percent-frozen ratios), averaged; the deterministic consensus where
    there are no ensembles; 10% when dry."""
    ests = []
    for ds in ("gefs", "ecens", "aifsens", "gfs", "aifs"):
        need = (f"{ds}_liq_mean", f"{ds}_frz_frac", f"{ds}_snow_mean")
        if all(c in X for c in need):
            ests.append((X[need[0]] * X[need[1]] / X[need[2]]).where(X[need[2]] > 0.05))
    if "cons_swe" in X and "cons_slr" in X:
        ests.append((X["cons_swe"] / X["cons_slr"]).where(X["cons_slr"] > 0.05))
    d = pd.concat(ests, axis=1).mean(axis=1) if ests else pd.Series(np.nan, index=X.index)
    d = d.clip(0.04, 0.25).fillna(0.10)
    dens = pd.Series(d.to_numpy(), index=pd.MultiIndex.from_frame(X[["station_id", "horizon_h"]]))
    idx = pd.MultiIndex.from_frame(pred[["station_id", "horizon_h"]])
    return pred.assign(dens=dens.reindex(idx).fillna(0.10).to_numpy())


def _tier1(X: pd.DataFrame, as_league: str, bundle: gbm.Bundle) -> pd.DataFrame:
    Xb = X.copy()
    Xb["league_code"] = NH_LEAGUES.index(as_league) if as_league in NH_LEAGUES else len(NH_LEAGUES)
    t1 = gbm.predict(bundle, Xb)
    t1["tier"] = "t1"
    has_ens = X.filter(like="_snow_q50").notna().any(axis=1).to_numpy()
    t1 = t1[has_ens]
    # beyond the training range a tree model can't extrapolate: lean on tier 0
    p99 = bundle.metrics.get("ref_snow_p99", {})
    lim = X["horizon_h"].map(lambda h: p99.get(str(int(h)), np.inf)).to_numpy()[has_ens]
    ext = X["ref_snow"].to_numpy()[has_ens] > lim
    if ext.any():
        t0 = fallback.predict_ens(X).set_index(["station_id", "horizon_h"])
        idx = pd.MultiIndex.from_frame(t1.loc[ext, ["station_id", "horizon_h"]])
        for c in (*QCOLS, "snowfall_in"):
            base = t0[c].reindex(idx).to_numpy()
            t1.loc[ext, c] = np.fmax(t1.loc[ext, c].to_numpy(), 0.5 * (t1.loc[ext, c].to_numpy() + base))
        t1.loc[ext, "tier"] = "t1x"
    return t1


def _stack(X: pd.DataFrame, as_league: str, sk: Stack) -> pd.DataFrame:
    Xb = X.copy()
    Xb["league_code"] = NH_LEAGUES.index(as_league) if as_league in NH_LEAGUES else len(NH_LEAGUES)
    cols: dict = {}
    for fm in sk.finals:
        cols.update(fm.predict(Xb))
    F = Xb[["station_id", "horizon_h"]].assign(league=as_league, **cols)
    p = stacker.predict(sk.stacker, F)
    out = pd.concat([Xb[list(KEYS)].reset_index(drop=True), p.reset_index(drop=True)], axis=1)
    out["prob_6in"] = np.clip(cols["p_ge6"], 0.001, 0.999) if "p_ge6" in cols else np.nan
    out["tier"] = "s1"
    has_ens = X.filter(like="_snow_q50").notna().any(axis=1).to_numpy()
    return out[has_ens]


def build(r: Round, network: bool = True, budget: float | None = None) -> Result:
    reg = stations(r.league)
    ids = [s for s in (r.stations if network else r.featured) if s in reg.index]
    feat_ids = [s for s in r.featured if s in reg.index]
    sk = load_stack(r.league)
    bname, bundle = (None, None) if sk else load_bundle(r.league)
    info: dict = {"round": r.key, "bundle": f"{sk.name} stack" if sk else bname}
    used = sk.name if sk else bname
    proxy = r.league if used == r.league else PROXY_LEAGUE.get(r.league, r.league)

    ens = live_ensembles(r, reg.loc[ids], info)
    if ens.empty:
        raise RuntimeError("no forecast inputs (every dynamical.org archive failed)")
    det = sibling_det(r, info)
    det = det.assign(round_date=r.round_date) if len(det) else det
    X = assemble(ens, det, r.league)

    t0 = fallback.predict_ens(X)
    t0["tier"] = "t0"
    parts = []
    if sk is not None:
        parts = [_stack(X, proxy, sk)]
    elif bundle is not None:
        parts = [_tier1(X, proxy, bundle)]
    pred = pd.concat([*parts, t0], ignore_index=True).drop_duplicates(["station_id", "horizon_h"], keep="first")
    tops = [c for c in X.columns if c.endswith(("_snow_q90", "_snow_mean", "_native_snow"))
            or c.startswith(("slr__", "snow__"))]
    pred = plausibility_cap(pred, X, tops)
    pred = _with_density(pred, X)
    # no post-processing layers yet: the model's output is what goes out
    pred["raw_in"] = pred["model_in"] = pred["snowfall_in"]

    df = sanitize(pred)
    raw = pred[["station_id", "horizon_h", "raw_in", "model_in", "snowfall_in", *QCOLS, "prob_6in", "tier", "dens"]]
    fe = set(feat_ids)
    info.update(
        rows=len(df),
        featured_coverage=round(df[df.station_id.isin(fe)].shape[0] / max(1, 3 * len(fe)), 3),
        tiers=raw["tier"].value_counts().to_dict(),
    )
    return Result(df=df, raw=raw, info=info, X=X[X["station_id"].isin(fe)].reset_index(drop=True))


def save_features(res: Result, path: Path, reason: str = "") -> None:
    """The feature rows behind a submission (featured stations), so a round
    can be re-predicted and compared with what the archives say the same
    forecast looked like. One file per round, replaced by each later pass like
    the history CSV; the pass and the model runs used are stored in it."""
    if res.X is None or res.X.empty:
        return
    X = res.X.copy()
    num = X.select_dtypes("number").columns.difference(["horizon_h"])
    X[num] = X[num].astype("float32").round(4)
    meta = {"pass_reason": reason, "built_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
            **{k: v for k, v in res.info.items() if k.endswith("_init")}}
    X = pd.concat([X, pd.DataFrame(meta, index=X.index)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(path, index=False, compression="zstd")


def sanitize(pred: pd.DataFrame) -> pd.DataFrame:
    """Exactly the PowderBench schema: numeric, clipped, monotone quantiles,
    p50 == snowfall_in, prob_6in only on 24 h rows, no duplicates."""
    df = pred[["station_id", "horizon_h", "snowfall_in", *QCOLS, "prob_6in"]].copy()
    dens = pred["dens"] if "dens" in pred else pd.Series(0.10, index=pred.index)
    df = df[df["horizon_h"].isin(HORIZONS)].dropna(subset=["snowfall_in"])
    df = df.drop_duplicates(["station_id", "horizon_h"])
    q = df[list(QCOLS)].apply(pd.to_numeric, errors="coerce")
    q["p50"] = df["snowfall_in"]
    for c in QCOLS:
        q[c] = q[c].fillna(q["p50"])
    arr = np.sort(np.clip(q.to_numpy(dtype=float), 0, 200), axis=1)
    df[list(QCOLS)] = np.round(arr, 2)
    df["snowfall_in"] = df["p50"]
    # cumulative horizons can't shrink: enforce h24 <= h48 <= h72 per station
    df = df.sort_values(["station_id", "horizon_h"])
    for c in (*QCOLS, "snowfall_in"):
        df[c] = df.groupby("station_id")[c].cummax()
    p = pd.to_numeric(df["prob_6in"], errors="coerce").clip(0.001, 0.999).round(3)
    df["prob_6in"] = p.where(df["horizon_h"] == 24)
    # water in the new snow (PowderBench's SWE / density tracks), cumulative too
    df["swe_in"] = (df["snowfall_in"] * dens.reindex(df.index).fillna(0.10)).round(2)
    df["swe_in"] = df.groupby("station_id")["swe_in"].cummax()
    return df.reset_index(drop=True)


def write(res: Result, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    res.df.to_csv(out, index=False)
    return out
