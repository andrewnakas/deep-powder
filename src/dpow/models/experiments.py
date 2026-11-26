"""Named variants of the median model, each judged by season CV against the
current settings on the same folds and rows (`experiment` command).

A variant is a function of the training table that returns what to change:
  df      a changed table (only when rows or columns must differ)
  extra   gbm.EXPERIMENTAL prefixes to let into the feature list
  drop    substrings of feature names to leave out
  params  LightGBM settings to override
  weight  name of a sample-weight column it added (starts with "_")
  resid   train on truth minus the raw consensus
  seeds   average this many models
Nothing here changes what `train` builds; a winner is moved into gbm.fit by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import features, store
from . import gbm

ADOPT_GAIN = 0.3  # pooled-mean gain needed, with every fold up
LEAGUE_SLACK = 0.3  # and no league down by more than this
BINS = ((-1, 1, "0-1"), (1, 3, "1-3"), (3, 6, "3-6"), (6, 1e9, "6+"))


def _w_featured(k: float):
    def f(df):
        df[f"_w_feat{k:g}"] = 1.0 + (k - 1.0) * df["featured"].to_numpy()
        return {"weight": f"_w_feat{k:g}"}
    return f


def _w_recent(df):
    """Each season back from the newest counts 0.8x the one after it."""
    age = df["season"].max() - df["season"]
    df["_w_recent"] = np.power(0.8, age.to_numpy(dtype=float))
    return {"weight": "_w_recent"}


def _model_cols(df: pd.DataFrame, model: str) -> list[str]:
    return [c for c in df.columns if c.endswith(f"__{model}")]


def _resummarize(d: pd.DataFrame) -> pd.DataFrame:
    features.summarize(d)
    if "ref_snow" in d:  # the ensemble entry's reference includes the deterministic consensus
        from ..dataset import add_ref

        add_ref(d)
    return d


def _moddrop(df):
    """A fifth of the multi-model rows again with one model blanked, so a
    model that is missing live is something the trees have seen. The copies
    are marked aug=1: trained on, never scored."""
    models = sorted({c.split("__", 1)[1] for c in df.columns if c.startswith("slr__")})
    pool = df[(df["nmod"] >= 3) & ((df["aug"] == 0) if "aug" in df else True)]
    sub = pool.sample(frac=0.2, random_state=3).copy()
    rng = np.random.default_rng(3)
    have = np.column_stack([sub[f"slr__{m}"].notna().to_numpy() for m in models])
    pick = np.argmax(np.where(have, rng.random(have.shape), -1.0), axis=1)
    for j, m in enumerate(models):
        sub.loc[pick == j, _model_cols(sub, m)] = np.nan
    sub = _resummarize(sub)
    sub["aug"] = 1
    out = pd.concat([df.assign(aug=df["aug"] if "aug" in df else 0), sub], ignore_index=True)
    return {"df": out}


def _without(model: str):
    def f(df):
        d = df.copy()
        d[_model_cols(d, model)] = np.nan
        return {"df": _resummarize(d)}
    return f


def _gfsx(drop: tuple[str, ...] = ()):
    """Hourly-GFS neighbourhood and timing features (nbhd.py), joined from the `nbhd` release."""
    def f(df):
        from .. import nbhd

        return {"df": nbhd.join(df), "extra": ("gfsx_",), "drop": tuple(f"gfsx_{d}" for d in drop)}
    return f


def _gfsx2(keep: tuple[str, ...]):
    """The second set of timing columns, on top of the adopted ones."""
    def f(df):
        from .. import nbhd

        return {"df": nbhd.join(df), "extra": tuple(f"gfsx_{k}" for k in keep)}
    return f


def _hrrx(drop: tuple[str, ...] = ()):
    """Hourly HRRR 24 h features (hrrx.py) on top of the adopted set."""
    def f(df):
        from .. import hrrx

        return {"df": hrrx.join(df), "extra": ("hrrx_",), "drop": tuple(f"hrrx_{d}" for d in drop)}
    return f


def _calib(keep: str = "all"):
    """Per-station, per-model ratios from earlier seasons (calib.py)."""
    def f(df):
        from .. import calib

        d = calib.add_history(df)
        drop = {"all": (), "adj": ("ratio__",), "ratio": ("adj__", "adj_mean", "adj_med")}[keep]
        return {"df": d, "drop": drop}
    return f


def _ana(adj: bool = True, with_calib: bool = False):
    """Long-record station ratios against the HRRR / GEFS analyses (ana.py),
    and the raw consensus scaled by each of them."""
    def f(df):
        from .. import ana, calib

        d = ana.join(df)
        if adj:
            for c in [c for c in d.columns if c.startswith("ana_ratio_")]:
                d[c.replace("ana_ratio_", "ana_adj_")] = (d["ref_snow"] * d[c]).astype("float32")
        if with_calib:
            d = calib.add_history(d)
        return {"df": d}
    return f


def _nbmx(drop: tuple[str, ...] = ()):
    """NOAA National Blend snowfall at the station (nbmx.py) on top of the adopted set."""
    def f(df):
        from .. import nbmx

        return {"df": nbmx.join(df), "extra": ("nbmx_",), "drop": tuple(f"nbmx_{d}" for d in drop)}
    return f


def _wndx(drop=()):
    def f(df):
        from .. import wndx

        return {"df": wndx.join(df), "extra": ("wndx_",), "drop": drop}
    return f



# Every column built from the Open-Meteo deterministic archive (featured
# stations only): the per-model columns, their summaries and the day splits.
OM_PREFIXES = ("slr__", "snow__", "prcp__", "twet__", "swe__", "cons_", "mean_slr", "max_slr", "min_slr", "sd_slr",
               "nsnow", "max_snow", "sd_snow", "nmod", "frac_snowy", "q25_slr", "q75_slr", "stitched")
ARCH_KEYS = ["station_id", "round_date", "horizon_h"]


def _blank(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """`cols` set to NaN (float32) on a copy-on-write copy: new memory for those columns only."""
    nan = pd.DataFrame(np.full((len(d), len(cols)), np.nan, np.float32), columns=cols, index=d.index)
    return pd.concat([d.drop(columns=cols), nan], axis=1)[list(d.columns)]


def _attach(d: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """w's columns joined onto d by the round keys, without copying d's own columns."""
    k = d[ARCH_KEYS].merge(w, on=ARCH_KEYS, how="left")
    d = d.copy(deep=False)
    for c in w.columns:
        if c not in ARCH_KEYS:
            d[c] = k[c].to_numpy()
    return d


def _no_om(df):
    """No Open-Meteo at all: what the model keeps from dynamical.org, NBM-free
    snowpack and static columns alone."""
    from ..dataset import add_ref

    d = _blank(df, [c for c in df.columns if c.startswith(OM_PREFIXES)])
    add_ref(d)
    return {"df": gbm.add_oof_ratio(d)}


def _ecens_old(df):
    """ECMWF ENS for 2020-10..2024-03 too (arch.py, the long Icechunk store):
    fills the ecens columns of rows that so far had GEFS/GFS/HRRR only."""
    from .. import arch
    from ..dataset import add_ref

    leagues = sorted(df["league"].unique())
    arch.pull(leagues, ["ecens"])
    d = df.copy(deep=False)
    cols = [c for c in d.columns if c.startswith("ecens_")]
    for lg in leagues:
        e = arch.load(lg, ["ecens"])
        if e.empty:
            continue
        m = (d["league"] == lg).to_numpy()
        got = d.loc[m, ARCH_KEYS].merge(e[ARCH_KEYS + [c for c in cols if c in e]], on=ARCH_KEYS, how="left")
        before = d.loc[m, "ecens_snow_q50"].notna().mean()
        for c in cols:
            if c in got:
                # in the table's own dtype: pandas refuses float64 values into a float32 column
                col = d[c].to_numpy(dtype="float32" if d[c].dtype.kind != "f" else None, copy=True)
                new = got[c].to_numpy(dtype=col.dtype, na_value=np.nan)
                col[m] = np.where(np.isnan(col[m]), new, col[m])
                d[c] = col
        print(f"  ecens_old {lg}: ECMWF ENS on {before:.0%} -> {d.loc[m, 'ecens_snow_q50'].notna().mean():.0%} "
              f"of rows", flush=True)
    add_ref(d)
    return {"df": gbm.add_oof_ratio(d)}


# harvested straight from the producer's open data: module -> the Open-Meteo model it stands in for
DIRECT_SOURCES = {"ukmo": "ukmo_seamless", "ecop": "ecmwf_ifs025",
                  "gfs_dyn": "gfs_seamless", "hrrr_dyn": "ncep_hrrr_conus"}  # dynom.py


def _direct_windows(name: str) -> pd.DataFrame:
    """The module's round windows: the precomputed asset when there is one
    (`<tag>__windows__12z.parquet`, built by `dpow direct-windows`), else built here."""
    import importlib

    ds = name.removesuffix("_dyn") if name.endswith("_dyn") else None
    if ds:
        from .. import dynom

        return dynom.load_windows(ds)
    mod = importlib.import_module(f"dpow.{name}")
    asset = f"{mod.TAG}__windows__12z.parquet"
    store.pull(mod.TAG, mod.DIR, asset, overwrite=True)
    if (mod.DIR / asset).exists():
        w = pd.read_parquet(mod.DIR / asset)
    else:
        mod.pull()
        w = mod.windows(12)
    w["round_date"] = pd.to_datetime(w["round_date"]).dt.date
    return w.astype({c: "float32" for c in w.columns if "__" in c})


def _direct(names: tuple[str, ...], mode: str):
    """Models read straight from their producers' open data (ukmo.py, ecop.py)
    for every station: `add` = their own model columns; `fill` = into the
    Open-Meteo model's columns where Open-Meteo gave none, consensus rebuilt;
    `no_om` = instead of every Open-Meteo column.

    The direct models have no snowfall field, so with `add`/`fill` most rows
    gain a model but no snowfall, and `gbm.snow_ready` would read that as an
    unfinished snowfall harvest and drop the whole snowfall family (12
    features: gfs_fill on 2026-11-23 lost them). Those variants keep the
    base table's decision."""
    def f(df):
        ready = gbm.snow_ready(df)
        d = _blank(df, [c for c in df.columns if c.startswith(OM_PREFIXES)]) if mode == "no_om" else df
        for name in names:
            w = _direct_windows(name)
            print(f"  {name}: {len(w)} round windows, {w['station_id'].nunique() if len(w) else 0} stations",
                  flush=True)
            new = [c for c in w.columns if "__" in c]
            d = _attach(d, w)
            print(f"  {name}: on {d[new[0]].notna().mean():.0%} of rows", flush=True)
            if mode == "fill":
                om = DIRECT_SOURCES[name]
                for c in new:
                    old = c.rsplit("__", 1)[0] + "__" + om
                    if old in d:
                        before = d[old].notna().mean()
                        d[old] = d[old].fillna(d[c])
                        if c == new[0]:
                            print(f"  {old}: {before:.0%} -> {d[old].notna().mean():.0%} of rows", flush=True)
                d = d.drop(columns=new)
        if mode == "fill":
            d = _resummarize(d)
        elif mode == "no_om":  # the consensus columns now come from the direct models alone
            d = gbm.add_oof_ratio(_resummarize(d))
        return {"df": d} if mode == "no_om" else {"df": d, "snow": ready}
    return f


def _ukmo(mode: str):
    return _direct(("ukmo",), mode)


# Open-Meteo models that also exist at the original source, free, with an archive:
# ECMWF open data (AWS), dynamical.org GFS/HRRR, the Met Office bucket (ukmo.py), NBM (AWS)
DIRECT_MODELS = ("ecmwf_ifs025", "gfs_seamless", "ukmo_seamless", "ncep_nbm_conus", "ncep_hrrr_conus")
# summary columns that features.summarize rebuilds from the per-model columns
REBUILT = features.SUMMARY_COLS


def _only(keep: tuple[str, ...]):
    """Open-Meteo's block with only the `keep` models: the others blanked, the
    consensus rebuilt from what is left, and the per-day/next/prev medians
    (which summarize cannot rebuild and were taken over every model) blanked."""
    def f(df):
        models = sorted({c.split("__", 1)[1] for c in df.columns if c.startswith("slr__")})
        gone = [m for m in models if m not in keep]
        cols = [c for m in gone for c in _model_cols(df, m)]
        cols += [c for c in df.columns if (c.startswith("cons_") and c not in REBUILT)
                 or c.startswith(("sd_slr_d", "max_slr_d"))]
        print(f"  only {keep}: blanking {gone} and {len(cols)} columns", flush=True)
        return {"df": gbm.add_oof_ratio(_resummarize(_blank(df, cols)))}
    return f


def _allst(frac: float):
    """Rows for `frac` of the stations the table leaves out (arch.py: every
    station of the league), trained on and never scored, so the scored rows
    stay the base's. Like the network sample, they have no Open-Meteo columns."""
    def f(df):
        from .. import arch, nbhd, truth, wndx
        from ..contract import stations
        from ..dataset import HF_LEAGUES, assemble, season_of

        leagues = [lg for lg in sorted(df["league"].unique()) if lg in HF_LEAGUES]
        arch.pull(leagues)
        parts = []
        for lg in leagues:
            have = set(df.loc[df["league"] == lg, "station_id"].unique())
            rest = sorted(set(stations(lg).index) - have)
            if not rest:
                continue
            pick = set(np.random.default_rng(5).choice(rest, max(1, int(len(rest) * frac)), replace=False))
            e = arch.load(lg)
            e = e[e["station_id"].isin(pick)]
            if e.empty:
                continue
            X = assemble(e, pd.DataFrame(), lg)
            r = X.merge(truth.windows(truth.hf_daily(lg, begin=min(X["round_date"]))), on=ARCH_KEYS, how="inner")
            r["league"], r["featured"] = lg, 0
            r["season"] = season_of(r["round_date"], lg)
            print(f"  allst {lg}: +{r['station_id'].nunique()} stations, +{len(r)} rows", flush=True)
            parts.append(r)
        if not parts:
            return {}
        extra = wndx.join(nbhd.join(pd.concat(parts, ignore_index=True)))
        extra = gbm.add_oof_ratio(extra)  # their own station ratios, out of season like the base's
        extra = extra[[c for c in df.columns if c in extra]]
        f64 = [c for c in extra.select_dtypes("float64").columns if c != "truth_in"]
        extra[f64] = extra[f64].astype("float32")
        base = df.assign(aug=df["aug"] if "aug" in df else 0)
        return {"df": pd.concat([base, extra.assign(aug=1)], ignore_index=True)}
    return f


def _new_rows(df):
    """The NEW_LEAGUES rows of the table release (every station, from arch.py),
    in the base table's columns and dtypes; None when there are none."""
    from .. import NEW_LEAGUES, store

    bundle = "nh"
    store.pull("table", store.TABLE, f"{bundle}__*.parquet")
    files = [p for p in sorted(store.TABLE.glob(f"{bundle}__*.parquet")) if p.stem.split("__")[1] in NEW_LEAGUES]
    if not files:
        print("  newlg: no new-league rows in the table release", flush=True)
        return None
    extra = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    extra["round_date"] = pd.to_datetime(extra["round_date"]).dt.date
    extra = gbm.add_oof_ratio(extra)  # their own station ratios, out of season like the base's
    extra = extra[[c for c in df.columns if c in extra]]
    f64 = [c for c in extra.select_dtypes("float64").columns if c != "truth_in"]
    extra[f64] = extra[f64].astype("float32")
    for lg, g in extra.groupby("league"):
        print(f"  newlg {lg}: {g['station_id'].nunique()} stations, {len(g)} rows, seasons "
              f"{sorted(g['season'].unique().tolist())}", flush=True)
    return extra


def _newlg(aug: int):
    """The new-league rows added to the base. aug=1: trained on, never scored,
    so the score is the base leagues' (does training on them help or hurt
    there?). aug=0: scored too, so the breakdown gives each new league's own
    season-CV accuracy."""
    def f(df):
        extra = _new_rows(df)
        if extra is None:
            return {}
        base = df.assign(aug=df["aug"] if "aug" in df else 0)
        return {"df": pd.concat([base, extra.assign(aug=aug)], ignore_index=True)}
    return f


def _newonly(df):
    """A model of the new leagues alone: trained and scored on their rows only,
    so its per-league numbers compare with newlg's (trained on all six). The
    pooled CV is theirs, not the base's: read the breakdown, not the delta.
    newlg cost the base leagues 0.73 (newlg_aug, every fold down, 11-25), so
    the new leagues would get their own model if it holds up alone."""
    extra = _new_rows(df)
    return {} if extra is None else {"df": extra.assign(aug=0)}


def _then(*steps):
    """Variants applied in turn, each to the table the previous one returned
    (a later step's spec keys win), e.g. a test on top of an adopted change."""
    def f(df):
        out = {}
        for step in steps:
            out.update(step(out.get("df", df)))
        return out
    return f


VARIANTS = {
    "ukmo_add": _ukmo("add"),
    "ukmo_fill": _ukmo("fill"),
    "no_om_ukmo": _ukmo("no_om"),
    "ecop_add": _direct(("ecop",), "add"),
    "ecop_fill": _direct(("ecop",), "fill"),
    "direct_fill": _direct(("ecop", "ukmo"), "fill"),
    "no_om_direct": _direct(("ecop", "ukmo"), "no_om"),
    "gfs_fill": _direct(("gfs_dyn",), "fill"),
    "hrrr_fill": _direct(("hrrr_dyn",), "fill"),
    "dyn_fill": _direct(("gfs_dyn", "hrrr_dyn"), "fill"),
    "gfs_add": _direct(("gfs_dyn",), "add"),
    # on top of the adopted GFS fill (gfs_fill +0.46): the ECMWF HRES and Met Office harvests
    "gfs_ecop_fill": _direct(("gfs_dyn", "ecop"), "fill"),
    "gfs_ukmo_fill": _direct(("gfs_dyn", "ukmo"), "fill"),
    "gfs_ecop_ukmo_fill": _direct(("gfs_dyn", "ecop", "ukmo"), "fill"),
    "direct_fill_all": _direct(("ecop", "ukmo", "gfs_dyn", "hrrr_dyn"), "fill"),
    "no_om_direct_all": _direct(("ecop", "ukmo", "gfs_dyn", "hrrr_dyn"), "no_om"),
    "no_om": _no_om,
    "only_direct": _only(DIRECT_MODELS),
    "only_ecmwf_gfs": _only(("ecmwf_ifs025", "gfs_seamless")),
    "ecens_old": _ecens_old,
    "newlg": _newlg(0),
    "newlg_aug": _newlg(1),
    "newonly": _newonly,
    "allst25": _allst(0.25),
    "allst50": _allst(0.5),
    # the same on top of the adopted GFS fill (compare with gfs_fill, 46.42); the extra stations' rows
    # get GFS from the fill too, as live stations will
    "gfs_ecens_old": _then(_ecens_old, _direct(("gfs_dyn",), "fill")),
    "gfs_allst25": _then(_allst(0.25), _direct(("gfs_dyn",), "fill")),
    "gfs_allst50": _then(_allst(0.5), _direct(("gfs_dyn",), "fill")),
    "hsp": lambda df: {"df": __import__("dpow.hsp", fromlist=["join"]).join(df), "extra": ("hsp_",)},
    # gefx (GEFS 31-member timing): -0.08, mean-only +0.10 on 2026-11-17; removed
    # wndx (storm wind + moisture): +0.52 every fold up, ADOPTED 2026-11-17; flow-only +0.10
    "ana": _ana(),
    "ana_ratio": _ana(adj=False),
    "ana_calib": _ana(with_calib=True),
    "calib": _calib(),
    "calib_adj": _calib("adj"),
    "calib_ratio": _calib("ratio"),
    "nbmx": _nbmx(),
    "nbmx_amount": _nbmx(drop=("last12", "pre12", "post6")),
    "hrrx": _hrrx(),
    "hrrx_amount": _hrrx(drop=("first12", "last12", "pre12", "post6", "peak3", "hours")),
    "gfsx2": _gfsx2(("pre6", "pre24", "post12", "first6", "last6", "last24", "liq_pre12", "rain", "t_first12",
                     "t_last12")),
    "gfsx2_edges": _gfsx2(("pre6", "pre24", "post12", "first6", "last6", "last24")),
    "gfsx2_phase": _gfsx2(("liq_pre12", "rain", "t_first12", "t_last12")),
    "gfsx": _gfsx(),
    "gfsx_nb": _gfsx(drop=("first12", "last12", "pre12", "post6", "peak6", "hours")),
    "gfsx_time": _gfsx(drop=("ctr", "max", "mean", "min", "sd", "liq_max", "frz_max", "tmin")),
    "w_feat2": _w_featured(2.0),
    "w_feat4": _w_featured(4.0),
    "w_feat8": _w_featured(8.0),
    "w_feat16": _w_featured(16.0),
    "w_feat1": _w_featured(1.0),
    "w_recent": _w_recent,
    "resid": lambda df: {"resid": True},
    "seedbag": lambda df: {"seeds": 3},
    "moddrop": _moddrop,
    "no_hrrr": _without("ncep_hrrr_conus"),
    "p_lr02": lambda df: {"params": {"learning_rate": 0.02}},
    "p_ff5": lambda df: {"params": {"feature_fraction": 0.5}},
    "p_l255": lambda df: {"params": {"num_leaves": 255, "min_child_samples": 150}},
    "no_days": lambda df: {"drop": ("cons_slr_d", "cons_prcp_d")},  # adopted 2026-11-14 (+1.15 CV in ope)
    "f_dayphys": lambda df: {"extra": ("cons_swe_d", "cons_twet_d")},
    "no_next": lambda df: {"drop": ("cons_slr_next", "cons_prcp_next")},  # adopted 2026-11-14 (+0.66 CV, 24 h +2.7)
    "f_prev": lambda df: {"extra": ("cons_slr_prev", "cons_prcp_prev")},
    "f_dayspread": lambda df: {"extra": ("sd_slr_d", "max_slr_d")},
    "f_spread": lambda df: {"extra": ("q25_slr", "q75_slr")},
    "f_tmin": lambda df: {"extra": ("cons_tmin",)},
    "f_all": lambda df: {"extra": gbm.EXPERIMENTAL},
}


def _pa(d: pd.DataFrame) -> float | None:
    d = d.dropna(subset=["climo_p50"])
    ref = float((d["climo_p50"] - d["truth_in"]).abs().sum())
    return round(100.0 * (1.0 - float((d["pred"] - d["truth_in"]).abs().sum()) / ref), 1) if ref > 0 else None


def breakdown(oof: pd.DataFrame) -> dict:
    """Pooled Powder Accuracy by league and horizon; 24 h MAE by truth amount."""
    from .. import NEW_LEAGUES

    out = {}
    if "league" in oof:
        out.update({f"{lg}": _pa(g) for lg, g in oof.groupby("league")})
        for lg, g in oof.groupby("league"):  # a new league next to the raw ensemble consensus on its rows
            if lg in NEW_LEAGUES and gbm.REF in g:
                out[f"{lg} raw"] = _pa(g.assign(pred=g[gbm.REF]).dropna(subset=["pred"]))
    out.update({f"{int(h)}h": _pa(g) for h, g in oof.groupby("horizon_h")})
    d = oof[oof["horizon_h"] == 24]
    err = (d["pred"] - d["truth_in"]).abs()
    for lo, hi, name in BINS:
        m = (d["truth_in"] > lo) & (d["truth_in"] <= hi) if lo >= 0 and lo != -1 else d["truth_in"] <= hi
        out[f"mae {name}"] = round(float(err[m].mean()), 2) if m.any() else None
    return out


def evaluate(df: pd.DataFrame, name: str, params: dict, folds: list) -> dict:
    spec = VARIANTS[name](df) if name != "base" else {}
    d = spec.get("df", df)
    feats = [f for f in gbm.feature_list(d, spec.get("extra", ()), snow=spec.get("snow"))
             if not any(x in f for x in spec.get("drop", ()))]
    oof: list = []
    r = gbm.cross_validate(d, feats, {**params, **spec.get("params", {})}, folds, weight=spec.get("weight"),
                           resid=spec.get("resid", False), seeds=spec.get("seeds", 1), oof=oof)
    r.update(name=name, n_features=len(feats), by=breakdown(pd.concat(oof, ignore_index=True)) if oof else {})
    print(f"  {name}: {r['powder_score']} per fold {r['per_fold']} {r['by']}", flush=True)
    return r


def verdict(base: dict, r: dict, leagues: list[str]) -> str:
    if r["powder_score"] is None or base["powder_score"] is None:
        return "no score"
    folds_up = all(a is not None and b is not None and a > b for a, b in zip(r["per_fold"], base["per_fold"]))
    lg_ok = all((r["by"].get(lg) or 0) >= (base["by"].get(lg) or 0) - LEAGUE_SLACK for lg in leagues)
    return "adopt" if r["powder_score"] - base["powder_score"] >= ADOPT_GAIN and folds_up and lg_ok else "no"


def run(df: pd.DataFrame, names: list[str], params: dict) -> str:
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; have {list(VARIANTS)}")
    df = gbm.add_oof_ratio(df)
    folds = gbm.cv_folds(df)
    if not folds:
        return "experiment: fewer than 3 full seasons, no CV"
    res = [evaluate(df, n, params, folds) for n in ["base", *names]]
    base, cols = res[0], list(res[0]["by"])
    leagues = sorted(df["league"].unique()) if "league" in df else []
    head = ["variant", "CV", "delta", *[str(f) for f in folds], *cols, "features", "verdict"]
    lines = [f"experiment on {len(df)} rows, folds {folds}, settings {params}", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in res:
        delta = "" if r is base or r["powder_score"] is None else f"{r['powder_score'] - base['powder_score']:+.2f}"
        cells = [r["name"], r["powder_score"], delta, *[None if x is None else round(x, 1) for x in r["per_fold"]],
                 *[r["by"].get(c) for c in cols], r["n_features"], "" if r is base else verdict(base, r, leagues)]
        lines.append("| " + " | ".join("" if c is None else str(c) for c in cells) + " |")
    return "\n".join(lines)
