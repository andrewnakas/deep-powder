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

OUT_Q = ("nn_q10", "nn_q25", "nn_q50", "nn_q75", "nn_q90")
OUT_WET = "nn_wet"
WET_IN = 0.05
HIDDEN = (512, 256, 128)
EMB = 16
EMB_DROP = 0.15
DROPOUT = 0.1
BATCH = 4096
LR = 1e-3
PCT_START = 0.1  # short warm-up: with 0.3 the rate was still rising when validation turned (2026-11-16)
MAX_EPOCHS = 20
PATIENCE = 4
WET_LOSS = 0.3


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class Scaler:
    """Signed log, then mean/sd of the training rows; NaN -> 0 with a mask."""

    def __init__(self, X: np.ndarray):
        Z = _slog(X)
        self.mu = np.nan_to_num(np.nanmean(Z, axis=0))
        sd = np.nan_to_num(np.nanstd(Z, axis=0))
        self.sd = np.where(sd > 1e-6, sd, 1.0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (_slog(X) - self.mu) / self.sd
        miss = np.isnan(Z)
        return np.concatenate([np.where(miss, 0.0, np.clip(Z, -8, 8)), miss], axis=1).astype(np.float32)


def _net(n_in: int, n_stations: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_stations + 1, EMB)  # 0 = unknown station
            layers, d = [], n_in + EMB
            for h in HIDDEN:
                layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(DROPOUT)]
                d = h
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(d, len(QUANTILES) + 1)

        def forward(self, x, sid):
            if self.training:  # so the unknown vector is a trained one
                sid = torch.where(torch.rand_like(sid, dtype=torch.float32) < EMB_DROP, torch.zeros_like(sid), sid)
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

    def predict(self, X: pd.DataFrame) -> dict:
        import torch

        x = X.reindex(columns=self.features).to_numpy(np.float32)
        Z = self.scaler(x)
        sid = X["station_id"].map(self.ids).fillna(0).to_numpy(np.int64)
        model = _net(Z.shape[1], len(self.ids))
        model.load_state_dict({k: torch.from_numpy(v) for k, v in self.state.items()})
        q, w = _forward(model, Z, sid)
        return {**{c: q[:, k] for k, c in enumerate(OUT_Q)}, OUT_WET: w}


def train(tr: pd.DataFrame, feats: list[str], weight=None, va: pd.DataFrame | None = None, va_weight=None,
          epochs: int | None = None, seed: int = 0, verbose: bool = True) -> Fitted:
    """With `va`: train up to MAX_EPOCHS, keep the epoch with the lowest
    weighted MAE of the median on it. With `epochs` (and no `va`): train that
    many, as the final fit on all seasons does."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = Scaler(tr[feats].to_numpy(np.float32))
    Xt = scaler(tr[feats].to_numpy(np.float32))
    ids = {s: i + 1 for i, s in enumerate(sorted(tr["station_id"].unique()))}
    st = tr["station_id"].map(ids).to_numpy(np.int64)
    yt = tr["truth_in"].to_numpy(np.float32)
    wt = np.ones(len(tr), np.float32) if weight is None else np.asarray(weight, np.float32)
    if va is not None:
        Xv = scaler(va[feats].to_numpy(np.float32))
        sv = va["station_id"].map(ids).fillna(0).to_numpy(np.int64)
        yv = va["truth_in"].to_numpy(np.float64)
        wv = np.ones(len(va)) if va_weight is None else np.asarray(va_weight, float)
    n_epochs = MAX_EPOCHS if va is not None else int(epochs or MAX_EPOCHS)

    model = _net(Xt.shape[1], len(ids))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    # the schedule is the same whatever the stopping epoch, so a final fit
    # for n epochs retraces the first n epochs of a validated run
    steps = MAX_EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=PCT_START)
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
            loss = ((pin + WET_LOSS * bce(w, tWet[b])) * tW[b]).sum() / tW[b].sum()
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
    return Fitted(list(feats), scaler, ids, state, best_epoch)


def fit_predict(tr: pd.DataFrame, va: pd.DataFrame, feats: list[str], weight=None, va_weight=None,
                seed: int = 0, verbose: bool = True, rounds: dict | None = None) -> dict:
    """Fit on `tr`, stop on `va` (weighted MAE of the median), predict `va`."""
    m = train(tr, feats, weight, va, va_weight, seed=seed, verbose=verbose)
    if rounds is not None:
        rounds.setdefault("epochs", []).append(m.epochs)
    return m.predict(va)
