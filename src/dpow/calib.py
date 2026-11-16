"""Per-station, per-model calibration from history.

The model already has one station factor (gbm.stn_ratio: truth over the raw
consensus). This adds one per forecast model: at every station, how much snow
that station reports for each inch a given model forecast. Models differ by
station (one model's grid point sits in a valley, another's on the ridge), so
a single factor cannot say which model to trust where.

For each model M and station S:
  ratio__M = (sum of truth + K) / (sum of M's snow + K)
over the 24 h rows where both exist, shrunk toward 1 by K inches of
pseudo-evidence and clipped. `adj__M` = M x ratio__M, and `adj_mean` /
`adj_med` summarise the calibrated models.

History only: a row of season s gets the ratios from seasons before s (the
first season with a model gets none), so training sees exactly what a live
forecast can know. Live forecasts use every season in the table.
"""

from __future__ import annotations

import pandas as pd

K = 4.0  # inches of pseudo-evidence toward a ratio of 1
CLIP = (0.25, 4.0)
MIN_ROWS = 10  # 24 h rows a station needs with a model before it gets a ratio


def model_columns(df: pd.DataFrame) -> list[str]:
    """Every per-model snowfall column in the table."""
    det = [c for c in df.columns if c.startswith("snow__")]
    ens = [c for c in df.columns if c.endswith(("_snow_mean", "_native_snow")) or c in ("gfsx_ctr", "hrrx_ctr")]
    return [*det, *ens, *(["cons_snow"] if "cons_snow" in df else [])]


def _ratios(h: pd.DataFrame, col: str) -> pd.Series:
    d = h.loc[h[col].notna() & h["truth_in"].notna(), ["station_id", "truth_in", col]]
    g = d.groupby("station_id").agg(t=("truth_in", "sum"), m=(col, "sum"), n=(col, "size"))
    g = g[g["n"] >= MIN_ROWS]
    return ((g["t"] + K) / (g["m"] + K)).clip(*CLIP)


def fit(df: pd.DataFrame, cols: list[str] | None = None) -> dict[str, pd.Series]:
    """Ratios from every row given (the live set)."""
    cols = cols or model_columns(df)
    h = df[(df["horizon_h"] == 24) & ((df["aug"] == 0) if "aug" in df else True)]
    return {c: _ratios(h, c) for c in cols}


def apply(df: pd.DataFrame, ratios: dict[str, pd.Series]) -> pd.DataFrame:
    new = {}
    for c, r in ratios.items():
        base = c.split("__", 1)[1] if c.startswith("snow__") else c
        f = df["station_id"].map(r).astype("float32")
        new[f"ratio__{base}"] = f
        new[f"adj__{base}"] = (df[c] * f).astype("float32")
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    adj = out[[k for k in new if k.startswith("adj__")]]
    out["adj_mean"] = adj.mean(axis=1).astype("float32")
    out["adj_med"] = adj.median(axis=1).astype("float32")
    return out


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Every row gets the ratios from the seasons before its own."""
    cols = model_columns(df)
    parts = []
    for s in sorted(df["season"].unique()):
        cur = df[df["season"] == s]
        past = df[df["season"] < s]
        parts.append(apply(cur, fit(past, cols) if len(past) else {c: pd.Series(dtype=float) for c in cols}))
    out = pd.concat(parts).loc[df.index]
    have = out[[c for c in out.columns if c.startswith("ratio__")]].notna().any(axis=1).mean()
    print(f"  calib: {len(cols)} models, {have:.0%} of rows have at least one station-model ratio", flush=True)
    return out
