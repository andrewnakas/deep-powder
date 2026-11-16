"""Level 2: turns the level-1 out-of-fold columns into the forecast.

Fitted per (league, horizon) cell, on featured rows, in three steps:

1. Blend. A convex combination of the level-1 point forecasts, by mean
   absolute error. Weight moves from one source to another in small steps
   while each move buys a real gain, starting from all weight on the anchor
   (the LightGBM median), and the result is shrunk toward the anchor when the
   cell has few round dates.
2. Dry rule. Rows whose chance of any snow is under a threshold go to zero.
   The threshold is chosen on the board score (mean of per-round scores),
   where a small forecast on a dry round is expensive, and is only accepted
   if the pooled score does not fall by more than POOLED_SLACK.
   (Out of fold the rule raises the pooled score too: +0.5, 2026-11-16.)
3. Intervals. The level-1 quantiles are re-centred on the blended median and
   their distance from it is scaled, per level, to the lowest pinball loss.

`evaluate` is nested: the stacker that predicts a fold season is fitted on the
other fold seasons only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import QCOLS, QUANTILES, metrics
from .blend import combine

ANCHOR = "q50"
POINT_SOURCES = ("q50", "mean", "q50_ens", "nn_q50", "ref_snow", "cons_snow", "l2")
WET_SOURCES = ("p_wet", "nn_wet")
QSRC = {0.1: "q10", 0.25: "q25", 0.5: "q50", 0.75: "q75", 0.9: "q90"}
STEP = 0.05
MIN_STEP_GAIN = 0.0  # a move only has to cut MAE (a 0.2% floor cost 0.8 pooled points out of fold, 2026-11-16)
ROUND_K = 30.0  # round dates of pseudo-evidence for "all weight on the anchor"
MIN_COVER = 0.9  # a source must exist on this share of a cell's rows
MIN_ROWS = 300
TAUS = tuple(np.round(np.arange(0.0, 0.91, 0.05), 2))
POOLED_SLACK = 0.1
BOARD_MIN_GAIN = 0.5  # board points a threshold must buy to be used


@dataclass
class Cell:
    weights: dict[str, float]
    wet: str | None = None
    tau: float = 0.0
    scale: dict[float, float] = field(default_factory=dict)
    n: int = 0


@dataclass
class Stacker:
    cells: dict[tuple[str, int], Cell]

    def describe(self) -> str:
        lines = ["| league | h | rows | blend | dry rule | interval scale |", "|---|---|---|---|---|---|"]
        for (lg, h), c in sorted(self.cells.items()):
            w = ", ".join(f"{k} {v:.2f}" for k, v in c.weights.items())
            dry = f"{c.wet} < {c.tau:.2f}" if c.wet and c.tau > 0 else "off"
            sc = ", ".join(f"{int(a * 100)}: {v:.2f}" for a, v in c.scale.items() if a != 0.5)
            lines.append(f"| {lg} | {h} | {c.n} | {w} | {dry} | {sc} |")
        return "\n".join(lines)


def _blend(X: np.ndarray, y: np.ndarray, n_rounds: int) -> np.ndarray:
    w = np.zeros(X.shape[1])
    w[0] = 1.0
    anchor = X[:, 0]

    def mae(t):
        p = combine(X, t)
        return float(np.abs(np.where(np.isnan(p), anchor, p) - y).mean())

    best = mae(w)
    while True:
        move = None
        for i in np.flatnonzero(w >= STEP - 1e-9):
            for j in range(len(w)):
                if i == j:
                    continue
                t = w.copy()
                t[i] -= STEP
                t[j] += STEP
                m = mae(t)
                if m < best * (1 - MIN_STEP_GAIN) and (move is None or m < move[0]):
                    move = (m, t)
        if move is None:
            break
        best, w = move
    k = n_rounds / (n_rounds + ROUND_K)
    w = k * w
    w[0] += 1 - k
    return w


def _point(g: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    cols = [c for c in weights if c in g]
    p = combine(g[cols].to_numpy(float), np.array([weights[c] for c in cols]))
    return np.where(np.isnan(p), g[ANCHOR].to_numpy(float), p)


def _dry(g: pd.DataFrame, point: np.ndarray) -> tuple[str | None, float]:
    """The wet-probability column and threshold that most improve the board score."""
    d = g.assign(_p=point)
    base_board, base_pooled = metrics.board(d, "_p"), metrics.pooled(d, "_p")
    best = (None, 0.0, base_board if base_board is not None else -np.inf)
    if base_board is None or base_pooled is None:
        return None, 0.0
    for col in WET_SOURCES:
        if col not in g or g[col].notna().mean() < MIN_COVER:
            continue
        pw = g[col].fillna(1.0).to_numpy()
        for tau in TAUS[1:]:
            d["_t"] = np.where(pw < tau, 0.0, point)
            b, p = metrics.board(d, "_t"), metrics.pooled(d, "_t")
            if p >= base_pooled - POOLED_SLACK and b > best[2] and b >= base_board + BOARD_MIN_GAIN:
                best = (col, float(tau), b)
    return best[0], best[1]


def _scale(g: pd.DataFrame, point: np.ndarray) -> dict[float, float]:
    """Per level, the multiplier on (quantile - median) with the lowest pinball
    loss around the final point forecast. 1.0 is in the grid, so on the fitting
    rows this never loses to the level-1 spread."""
    out = {}
    y, mid = g["truth_in"].to_numpy(), g[ANCHOR].to_numpy()
    grid = np.round(np.arange(0.4, 2.01, 0.1), 2)
    for a in QUANTILES:
        col = QSRC[a]
        if a == 0.5 or col not in g or g[col].isna().all():
            continue
        dist = np.nan_to_num(g[col].to_numpy() - mid)
        dist = np.minimum(dist, 0.0) if a < 0.5 else np.maximum(dist, 0.0)
        loss = []
        for s in grid:
            d = y - np.clip(point + s * dist, 0, None)
            loss.append(np.maximum(a * d, (a - 1) * d).mean())
        out[a] = float(grid[int(np.argmin(loss))])
    return out


# A learned second level: a small LightGBM median over the tree learners'
# outputs. It can mix them by situation (lean on the mean when the upper
# quantiles are far out, on the classifiers near zero), which fixed weights
# cannot. It joins the blend as one more source, `l2`. Out of fold on
# 2026-11-16 it was worth +0.1 pooled, +1.0 at 24 h and -0.2 in. MAE on 6+ in. days.
L2_INPUTS = ("q10", "q25", "q50", "q75", "q90", "mean", "p_wet", "p_ge1", "p_ge6", "horizon_h", "climo_p50",
             "ref_snow")
L2_LEAGUES = ("alps", "observers", "stations")
L2_PARAMS = dict(objective="quantile", alpha=0.5, learning_rate=0.05, num_leaves=31, min_child_samples=300,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
L2_ROUNDS = 300
L2_FEATURED_WEIGHT = 4.0


def l2_matrix(df: pd.DataFrame) -> pd.DataFrame:
    F = df.reindex(columns=list(L2_INPUTS)).astype(float)
    F["league_code"] = df["league"].map({lg: i for i, lg in enumerate(L2_LEAGUES)}).to_numpy(float)
    return F


def l2_fit(df: pd.DataFrame):
    import lightgbm as lgb

    d = df.dropna(subset=[ANCHOR, "truth_in"])
    w = 1.0 + (L2_FEATURED_WEIGHT - 1.0) * d["featured"].to_numpy(float) if "featured" in d else None
    return lgb.train(L2_PARAMS, lgb.Dataset(l2_matrix(d), d["truth_in"], weight=w), L2_ROUNDS)


def l2_predict(model, df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(l2_matrix(df)), 0.0, 200.0)


def l2_oof(df: pd.DataFrame) -> np.ndarray:
    """`l2` for every row, each fold predicted by a model fitted on the others."""
    out = np.full(len(df), np.nan)
    for s in sorted(df["fold"].dropna().unique()):
        va = (df["fold"] == s).to_numpy()
        tr = df[~va & df["fold"].notna().to_numpy()]
        if tr.empty:
            continue
        out[va] = l2_predict(l2_fit(tr), df[va])
    return out


def fit(df: pd.DataFrame) -> Stacker:
    d = metrics.scored_rows(df).dropna(subset=[ANCHOR])
    cells = {}
    for (lg, h), g in d.groupby(["league", "horizon_h"]):
        if len(g) < MIN_ROWS:
            continue
        src = [ANCHOR] + [c for c in POINT_SOURCES if c != ANCHOR and c in g and g[c].notna().mean() >= MIN_COVER]
        w = _blend(g[src].to_numpy(float), g["truth_in"].to_numpy(float), g["round_date"].nunique())
        weights = {c: round(float(v), 4) for c, v in zip(src, w) if v > 1e-9}
        point = _point(g, weights)
        wet, tau = _dry(g, point)
        if wet:
            point = np.where(g[wet].fillna(1.0).to_numpy() < tau, 0.0, point)
        cells[(str(lg), int(h))] = Cell(weights, wet, tau, _scale(g, point), len(g))
    return Stacker(cells)


def predict(st: Stacker, df: pd.DataFrame) -> pd.DataFrame:
    """snowfall_in and p10..p90 for every row (cells never fitted pass the anchor through)."""
    out = pd.DataFrame(index=df.index, columns=["snowfall_in", *QCOLS], dtype=float)
    for (lg, h), g in df.groupby(["league", "horizon_h"]):
        c = st.cells.get((str(lg), int(h)), Cell({ANCHOR: 1.0}))
        point = _point(g, c.weights)
        dry = g[c.wet].fillna(1.0).to_numpy() < c.tau if c.wet and c.tau > 0 else np.zeros(len(g), bool)
        point = np.where(dry, 0.0, point)
        mid = g[ANCHOR].to_numpy(float)
        q = np.empty((len(g), len(QUANTILES)))
        for k, a in enumerate(QUANTILES):
            col = QSRC[a]
            if a == 0.5 or col not in g:
                q[:, k] = point
                continue
            dist = np.nan_to_num(g[col].to_numpy(float) - mid) * c.scale.get(a, 1.0)
            dist = np.minimum(dist, 0.0) if a < 0.5 else np.maximum(dist, 0.0)
            q[:, k] = point + dist
        q = np.clip(q, 0.0, 200.0)
        q[:, 1] = np.minimum(q[:, 1], q[:, 2])
        q[:, 0] = np.minimum(q[:, 0], q[:, 1])
        q[:, 3] = np.maximum(q[:, 3], q[:, 2])
        q[:, 4] = np.maximum(q[:, 4], q[:, 3])
        out.loc[g.index, "snowfall_in"] = q[:, 2]
        out.loc[g.index, list(QCOLS)] = q
    return out


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Stacker]]:
    """Nested: each fold season is predicted by a stacker fitted on the others.
    Returns the frame with `stack` and `stack_p10..p90` columns."""
    out = df.copy()
    for c in ("stack", *(f"stack_{q}" for q in QCOLS)):
        out[c] = np.nan
    fitted = []
    for s in sorted(df["fold"].dropna().unique()):
        is_va = (df["fold"] == s).to_numpy()
        if is_va.all():
            break
        st = fit(df[~is_va & df["fold"].notna().to_numpy()])
        p = predict(st, df[is_va])
        out.loc[is_va, "stack"] = p["snowfall_in"].to_numpy()
        for q in QCOLS:
            out.loc[is_va, f"stack_{q}"] = p[q].to_numpy()
        fitted.append(st)
    return out, fitted
