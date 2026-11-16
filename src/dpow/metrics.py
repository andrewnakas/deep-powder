"""Scores the way PowderBench's boards do, for out-of-fold predictions.

Two headline numbers, both on featured rows with a climatology forecast:

- `board`: what the leaderboard shows today. Each round (league, round date)
  gets 100 * (1 - MAE / MAE_climatology); a round whose climatology error is
  zero has no score; the league figure is the mean over its rounds. Dry rounds
  have a tiny denominator, so a small forecast on a dry day costs a lot.
- `pooled`: 100 * (1 - sum|err| / sum|climatology err|) over all rows, the
  rule proposed for the boards (PowderBench PR #575).

A frame here has station_id, round_date, horizon_h, league, truth_in,
climo_p50 and one column per forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import POWDER_IN, QCOLS, QUANTILES

BINS = ((-1.0, 1.0, "0-1"), (1.0, 3.0, "1-3"), (3.0, 6.0, "3-6"), (6.0, 1e9, "6+"))


def scored_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Featured rows with climatology: the rows the headline board uses."""
    d = df.dropna(subset=["climo_p50", "truth_in"])
    if "aug" in d:
        d = d[d["aug"] == 0]
    if "featured" in d and d["featured"].any():
        d = d[d["featured"] == 1]
    return d


def pooled(d: pd.DataFrame, col: str) -> float | None:
    g = d.dropna(subset=[col])
    ref = float((g["climo_p50"] - g["truth_in"]).abs().sum())
    return 100.0 * (1.0 - float((g[col] - g["truth_in"]).abs().sum()) / ref) if ref > 0 else None


def round_scores(d: pd.DataFrame, col: str) -> pd.Series:
    """Score of every (league, round_date) round that has one."""
    g = d.dropna(subset=[col])
    err = (g[col] - g["truth_in"]).abs()
    ref = (g["climo_p50"] - g["truth_in"]).abs()
    by = [g["league"], g["round_date"]] if "league" in g else [g["round_date"]]
    s = pd.DataFrame({"e": err, "r": ref}).groupby(by).sum()
    s = s[s["r"] > 0]
    return 100.0 * (1.0 - s["e"] / s["r"])


def board(d: pd.DataFrame, col: str) -> float | None:
    """Mean of per-round scores; with several leagues, the mean of the league means."""
    s = round_scores(d, col)
    if s.empty:
        return None
    return float(s.groupby(level=0).mean().mean()) if s.index.nlevels > 1 else float(s.mean())


def pinball(d: pd.DataFrame, cols: tuple[str, ...] = QCOLS) -> float | None:
    if not all(c in d for c in cols):
        return None
    g = d.dropna(subset=list(cols))
    if g.empty:
        return None
    y = g["truth_in"].to_numpy()
    loss = [np.maximum(a * (y - g[c].to_numpy()), (a - 1) * (y - g[c].to_numpy())).mean()
            for a, c in zip(QUANTILES, cols)]
    return float(np.mean(loss))


def brier6(d: pd.DataFrame, col: str) -> float | None:
    g = d[d["horizon_h"] == 24].dropna(subset=[col]) if col in d else d.iloc[:0]
    if g.empty:
        return None
    return float(((g[col] - (g["truth_in"] >= POWDER_IN)) ** 2).mean())


def _r(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def summary(df: pd.DataFrame, col: str) -> dict:
    """Everything one forecast column is judged on (already-filtered frames are fine)."""
    d = scored_rows(df).dropna(subset=[col])
    out = {"n": int(len(d)), "pooled": _r(pooled(d, col), 2), "board": _r(board(d, col), 2)}
    if "league" in d:
        for lg, g in d.groupby("league"):
            out[f"{lg} pooled"] = _r(pooled(g, col))
            out[f"{lg} board"] = _r(board(g, col))
    for h, g in d.groupby("horizon_h"):
        out[f"{int(h)}h"] = _r(pooled(g, col))
    h24 = d[d["horizon_h"] == 24]
    err = (h24[col] - h24["truth_in"]).abs()
    for lo, hi, name in BINS:
        m = (h24["truth_in"] > lo) & (h24["truth_in"] <= hi)
        out[f"mae {name}"] = _r(err[m].mean(), 2) if m.any() else None
    return out


def table(df: pd.DataFrame, cols: list[str], base: str | None = None) -> str:
    """Markdown comparison of forecast columns on the rows where all of them exist."""
    d = scored_rows(df).dropna(subset=cols)
    rows = {c: summary(d, c) for c in cols}
    keys = [k for k in rows[cols[0]] if k != "n"]
    head = ["forecast", *keys]
    lines = [f"{len(d)} featured rows, {d['round_date'].nunique()} round dates", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in cols:
        name = f"**{c}**" if c == base else c
        lines.append("| " + " | ".join([name, *("" if rows[c].get(k) is None else str(rows[c][k]) for k in keys)])
                     + " |")
    return "\n".join(lines)
