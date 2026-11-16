"""Level-1 learners and their out-of-fold predictions.

Every learner is run the same way: each fold season is held out in turn, the
learner is fitted on the other seasons and predicts every row of the held-out
one (featured and network). The stacker (stacker.py) is fitted on these
out-of-fold columns, and every comparison between learners reads them, so
"better" always means on the same rows and folds.

Learners (columns they write):
  q50      LightGBM median, all features                  -> q50
  qlo      LightGBM 10% and 25% quantiles                 -> q10, q25
  qhi      LightGBM 75% and 90% quantiles                 -> q75, q90
  hurdle   LightGBM classifiers for >0, >=1 and >=6 in.   -> p_wet, p_ge1, p_ge6
  tweedie  LightGBM mean (Tweedie loss)                   -> mean
  q50_ens  median without the Open-Meteo block            -> q50_ens
  nn       neural net trained off Actions (models/net.py) -> nn_q10..nn_q90, nn_wet

An out-of-fold frame has the round keys, `league`, `fold` and the learner's
columns; `oof/<learner>.parquet` in the `oof` release.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import gbm

KEYS = ["league", "station_id", "round_date", "horizon_h"]
# Columns of the table that the stacker and the reports need next to the predictions.
META = [*KEYS, "season", "featured", "truth_in", "climo_p50", "climo_p6", "ref_snow", "cons_snow"]
DET_PREFIXES = ("cons_", "mean_slr", "max_slr", "min_slr", "sd_slr", "nmod", "frac_snowy", "nsnow", "max_snow",
                "sd_snow", "q25_slr", "q75_slr", "stitched", "ref_snow")
WET_IN = 0.05  # a row counts as snow when truth is above this


def det_column(c: str) -> bool:
    """A column that only exists when the Open-Meteo multi-model block does."""
    return "__" in c or c.startswith(DET_PREFIXES)


@dataclass(frozen=True)
class Spec:
    kind: str  # quantile | binary | tweedie | net
    outputs: tuple[tuple[str, float], ...]  # (column, alpha or threshold)
    ens_only: bool = False


LEARNERS: dict[str, Spec] = {
    "q50": Spec("quantile", (("q50", 0.5),)),
    "qlo": Spec("quantile", (("q10", 0.1), ("q25", 0.25))),
    "qhi": Spec("quantile", (("q75", 0.75), ("q90", 0.9))),
    "hurdle": Spec("binary", (("p_wet", WET_IN), ("p_ge1", 1.0), ("p_ge6", 6.0))),
    "tweedie": Spec("tweedie", (("mean", 1.3),)),
    "q50_ens": Spec("quantile", (("q50_ens", 0.5),), ens_only=True),
    "nn": Spec("net", ()),
}


def features_for(df: pd.DataFrame, spec: Spec) -> list[str]:
    feats = gbm.feature_list(df)
    return [f for f in feats if not det_column(f)] if spec.ens_only else feats


def _objective(spec: Spec, value: float) -> dict:
    if spec.kind == "quantile":
        return {"objective": "quantile", "alpha": value}
    if spec.kind == "binary":
        return {"objective": "binary"}
    return {"objective": "tweedie", "tweedie_variance_power": value, "metric": "l1"}


def _label(spec: Spec, y: pd.Series, value: float):
    return (y > value if value == WET_IN else y >= value).astype(int) if spec.kind == "binary" else y


def fit_predict(tr: pd.DataFrame, va: pd.DataFrame, feats: list[str], spec: Spec, params: dict) -> dict:
    """Fit on `tr` (early stopping on `va`, as the season CV always has) and
    predict `va`: column -> array."""
    w = gbm.row_weight(tr)
    out, ds, dv = {}, None, None
    for col, value in spec.outputs:
        if ds is None or spec.kind == "binary":  # one binned dataset serves every quantile level
            ds = gbm._data(tr[feats], _label(spec, tr["truth_in"], value), weight=w)
            dv = gbm._data(va[feats], _label(spec, va["truth_in"], value), ds, weight=gbm.row_weight(va))
        m, _ = gbm._train(ds, _objective(spec, value), dv, params=params)
        p = m.predict(va[feats], num_iteration=m.best_iteration or None)
        out[col] = np.clip(p, 0.0, 1.0) if spec.kind == "binary" else np.clip(p, 0.0, 200.0)
    return out


def oof(df: pd.DataFrame, name: str, params: dict | None = None, folds: list | None = None) -> pd.DataFrame:
    """Out-of-fold predictions of one learner for every row of the fold seasons."""
    spec = LEARNERS[name]
    params = params or gbm.REFERENCE
    if "stn_ratio" not in df:
        df = gbm.add_oof_ratio(df)
    folds = folds if folds is not None else gbm.cv_folds(df)
    feats = features_for(df, spec)
    parts = []
    for s in folds:
        is_va = (df["season"] == s).to_numpy()
        tr = df.loc[~is_va, [*feats, "truth_in", "featured", "station_id"]]
        va = df.loc[is_va]
        if tr.empty or va.empty:
            continue
        if spec.kind == "net":
            from . import net

            pred = net.fit_predict(tr, va, feats, weight=gbm.row_weight(tr), va_weight=gbm.row_weight(va))
        else:
            pred = fit_predict(tr, va, feats, spec, params)
        del tr
        parts.append(va[KEYS].assign(fold=str(s), **pred))
        print(f"  {name} fold {s}: {len(va)} rows, {len(feats)} features", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*KEYS, "fold"])


def join(meta: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Table columns (META) with every learner's out-of-fold columns beside
    them, on the rows the first learner covers."""
    out = None
    for f in frames:
        f = f.copy()
        f["round_date"] = pd.to_datetime(f["round_date"]).dt.date
        out = f if out is None else out.merge(f.drop(columns="fold", errors="ignore"), on=KEYS, how="left")
    m = meta[[c for c in META if c in meta]].copy()
    m["round_date"] = pd.to_datetime(m["round_date"]).dt.date
    return out.merge(m, on=KEYS, how="left")
