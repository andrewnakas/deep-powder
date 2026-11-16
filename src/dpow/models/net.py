"""Level-1 neural net (CPU-sized): an MLP with a station embedding.

Why it earns a place next to the trees: it is smooth where trees are
piecewise constant, it learns a per-station vector instead of one ratio, and
its errors differ from LightGBM's, which is what a blend needs.

Inputs: every model feature, signed-log scaled and standardised on the
training rows, missing values as 0 plus a 0/1 "missing" input per feature.
Outputs: the five quantiles of log1p(snowfall) (pinball loss; quantiles
survive the monotone transform) and the chance of any snow (cross-entropy).
The station embedding is dropped at random during training so that a station
never seen before gets the shared "unknown" vector at prediction time.

torch is an optional dependency (`pip install -e '.[nn]'`), imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import QUANTILES

QNAMES = ("q10", "q25", "q50", "q75", "q90")
WET_IN = 0.05
BATCH = 4096
MAX_EPOCHS = 20
PATIENCE = 4
HIDDEN = (512, 256, 128)
# Settings a variant may override (learners.Spec.net)
# Out of fold on 2026-11-16 (pooled, alone): lr 1e-3 / dropout 0.1 41.86; lr 3e-4 / dropout 0.2 43.03
# (these); heavier regularisation 42.76; no station embedding 37.62. In the stack the choice moves
# the score by about 0.1, and averaging three nets by another 0.06: not worth three times the job.
DEFAULTS = dict(emb=16, emb_drop=0.15, dropout=0.2, lr=3e-4, weight_decay=1e-5, wet_loss=0.3,
                pct_start=0.1)  # short warm-up: with 0.3 the rate was still rising when validation turned
OUT_Q = tuple(f"nn_{q}" for q in QNAMES)
OUT_WET = "nn_wet"


def out_cols(prefix: str = "nn") -> tuple[tuple[str, ...], str]:
    return tuple(f"{prefix}_{q}" for q in QNAMES), f"{prefix}_wet"


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class Scaler:
    """Signed log, then mean/sd of the training rows; NaN -> 0 with a 0/1
    "missing" input per feature. Works column by column: a second full copy
    of a 2.7M-row table is what took the training job to the runner's limit."""

    def __init__(self, df: pd.DataFrame, feats: list[str]):
        mu, sd = [], []
        for c in feats:
            z = _slog(df[c].to_numpy(np.float32))
            mu.append(np.nanmean(z) if np.isfinite(z).any() else 0.0)
            v = np.nanstd(z) if np.isfinite(z).any() else 0.0
            sd.append(v if v > 1e-6 else 1.0)
        self.mu, self.sd = np.asarray(mu, np.float32), np.asarray(sd, np.float32)

    def __call__(self, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
        n = len(feats)
        out = np.zeros((len(df), 2 * n), np.float32)
        for j, c in enumerate(feats):
            z = df[c].to_numpy(np.float32) if c in df else np.full(len(df), np.nan, np.float32)
            z = (_slog(z) - self.mu[j]) / self.sd[j]
            miss = np.isnan(z)
            out[:, j] = np.where(miss, 0.0, np.clip(z, -8, 8))
            out[:, n + j] = miss
        return out


def _net(n_in: int, n_stations: int, cfg: dict):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_stations + 1, cfg["emb"])  # 0 = unknown station
            layers, d = [], n_in + cfg["emb"]
            for h in HIDDEN:
                layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(cfg["dropout"])]
                d = h
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(d, len(QUANTILES) + 1)

        def forward(self, x, sid):
            if self.training:  # so the unknown vector is a trained one
                drop = torch.rand_like(sid, dtype=torch.float32) < cfg["emb_drop"]
                sid = torch.where(drop, torch.zeros_like(sid), sid)
            out = self.head(self.body(torch.cat([x, self.emb(sid)], dim=1)))
            return out[:, :-1], out[:, -1]

    return Net()


def _forward(model, X: np.ndarray, sid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    qs, ws = [], []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            q, w = model(torch.from_numpy(X[i:i + 65536]), torch.from_numpy(sid[i:i + 65536]))
            qs.append(q.numpy())
            ws.append(torch.sigmoid(w).numpy())
    q = np.sort(np.clip(np.expm1(np.concatenate(qs)), 0.0, 200.0), axis=1)
    return q, np.concatenate(ws)


@dataclass
class Fitted:
    """Everything prediction needs, picklable without torch objects."""
    features: list[str]
    scaler: Scaler
    ids: dict
    state: dict  # name -> numpy array
    epochs: int = 0
    cfg: dict | None = None
    prefix: str = "nn"

    def predict(self, X: pd.DataFrame) -> dict:
        import torch

        Z = self.scaler(X, self.features)
        sid = X["station_id"].map(self.ids).fillna(0).to_numpy(np.int64)
        model = _net(Z.shape[1], len(self.ids), {**DEFAULTS, **(self.cfg or {})})
        model.load_state_dict({k: torch.from_numpy(v) for k, v in self.state.items()})
        q, w = _forward(model, Z, sid)
        qc, wc = out_cols(self.prefix)
        return {**{c: q[:, k] for k, c in enumerate(qc)}, wc: w}


def train(tr: pd.DataFrame, feats: list[str], weight=None, va: pd.DataFrame | None = None, va_weight=None,
          epochs: int | None = None, seed: int = 0, verbose: bool = True, cfg: dict | None = None,
          prefix: str = "nn") -> Fitted:
    """With `va`: train up to MAX_EPOCHS, keep the epoch with the lowest
    weighted MAE of the median on it. With `epochs` (and no `va`): train that
    many, as the final fit on all seasons does."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    c = {**DEFAULTS, **(cfg or {})}
    scaler = Scaler(tr, feats)
    Xt = scaler(tr, feats)
    ids = {s: i + 1 for i, s in enumerate(sorted(tr["station_id"].unique()))}
    st = tr["station_id"].map(ids).to_numpy(np.int64)
    yt = tr["truth_in"].to_numpy(np.float32)
    wt = np.ones(len(tr), np.float32) if weight is None else np.asarray(weight, np.float32)
    if va is not None:
        Xv = scaler(va, feats)
        sv = va["station_id"].map(ids).fillna(0).to_numpy(np.int64)
        yv = va["truth_in"].to_numpy(np.float64)
        wv = np.ones(len(va)) if va_weight is None else np.asarray(va_weight, float)
    n_epochs = MAX_EPOCHS if va is not None else int(epochs or MAX_EPOCHS)

    model = _net(Xt.shape[1], len(ids), c)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    # the schedule is the same whatever the stopping epoch, so a final fit
    # for n epochs retraces the first n epochs of a validated run
    steps = MAX_EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps, pct_start=c["pct_start"])
    alphas = torch.tensor(QUANTILES, dtype=torch.float32)
    tX, tS = torch.from_numpy(Xt), torch.from_numpy(st)
    tY, tW = torch.from_numpy(np.log1p(yt)), torch.from_numpy(wt)
    tWet = torch.from_numpy((yt > WET_IN).astype(np.float32))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    best, best_state, best_epoch, bad = np.inf, None, n_epochs, 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(tr)))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            q, w = model(tX[b], tS[b])
            d = tY[b].unsqueeze(1) - q
            pin = torch.maximum(alphas * d, (alphas - 1) * d).mean(dim=1)
            loss = ((pin + c["wet_loss"] * bce(w, tWet[b])) * tW[b]).sum() / tW[b].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        if va is None:
            continue
        q, _ = _forward(model, Xv, sv)
        mae = float((np.abs(q[:, 2] - yv) * wv).sum() / wv.sum())
        if verbose:
            print(f"    nn epoch {epoch + 1}: validation MAE {mae:.4f}", flush=True)
        if mae < best - 1e-4:
            best, bad, best_epoch = mae, 0, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    return Fitted(list(feats), scaler, ids, state, best_epoch, cfg, prefix)


def fit_predict(tr: pd.DataFrame, va: pd.DataFrame, feats: list[str], weight=None, va_weight=None,
                seed: int = 0, verbose: bool = True, rounds: dict | None = None, cfg: dict | None = None,
                prefix: str = "nn") -> dict:
    """Fit on `tr`, stop on `va` (weighted MAE of the median), predict `va`."""
    m = train(tr, feats, weight, va, va_weight, seed=seed, verbose=verbose, cfg=cfg, prefix=prefix)
    if rounds is not None:
        rounds.setdefault("epochs", []).append(m.epochs)
    return m.predict(va)
