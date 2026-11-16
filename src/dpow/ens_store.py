"""Reader for the ensemble round features harvested by open-powder-ensemble.

A unit = (dataset, league or `league-net`, month), one parquet in the
sibling repo's `ens-store*` releases, rows keyed (station_id, round_date,
horizon_h). `-net` units hold a fixed sample of non-featured stations.
"""

from __future__ import annotations

import pandas as pd

from . import WORK
from .sources.dynamical import DATASETS

ENS_STORE = WORK / "ens_store"
NET = "-net"


def load(league: str) -> pd.DataFrame:
    """Every dataset's harvested features for a league, joined on the round keys."""
    keys = ["station_id", "round_date", "horizon_h"]
    out = None
    for ds in DATASETS:
        tag = f"{ds}{DATASETS[ds].tag}"
        files = (sorted(ENS_STORE.glob(f"ens__{tag}__{league}__*.parquet"))
                 + sorted(ENS_STORE.glob(f"ens__{tag}__{league}{NET}__*.parquet")))
        if not files:
            continue
        f = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True)
        f["round_date"] = pd.to_datetime(f["round_date"]).dt.date
        f = f.drop_duplicates(keys)
        out = f if out is None else out.merge(f, on=keys, how="outer")
    return out if out is not None else pd.DataFrame(columns=keys)
