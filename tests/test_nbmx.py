"""NBM harvester: .idx parsing, row order, windows. Offline (no S3)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dpow import nbmx

IDX = """\
61:71000000:d=2025121512:APCP:surface:18-24 hour acc fcst:
62:72110631:d=2025121512:ASNOW:surface:23-24 hour acc fcst:
63:72685814:d=2025121512:ASNOW:surface:18-24 hour acc fcst:
64:73300000:d=2025121512:ASNOW:surface:18-24 hour acc@(fcst,dt=6 hour),missing=0:50% level
65:73900000:d=2025121512:SNOWLR:surface:24 hour fcst:
"""


def test_byte_range_picks_the_deterministic_6h_bucket():
    assert nbmx.byte_range(IDX, 24) == (72685814, 73299999)
    assert nbmx.byte_range(IDX, 30) is None


def test_byte_range_last_message_is_open_ended():
    assert nbmx.byte_range(IDX.splitlines()[2] + "\n", 24) == (72685814, None)


def test_init_before_cutoff_minus_latency(monkeypatch):
    class L:
        def cutoff_utc(self, d):
            return pd.Timestamp(d) - pd.Timedelta(hours=1)  # 23:00 UTC the day before

    import powderbench.leagues as lg

    monkeypatch.setattr(lg, "get_league", lambda name: L())
    # latest usable = 23:00 - 2 h margin - 2 h latency = 19:00 -> the 18z run
    assert nbmx.init_for_round(date(2025, 12, 16), "stations") == pd.Timestamp("2025-12-15 18:00")


def test_features_sum_the_windows(monkeypatch):
    st = pd.DataFrame({"latitude": [40.0], "longitude": [-111.0]}, index=pd.Index(["a"], name="station_id"))
    init = pd.Timestamp("2025-12-15 12:00")
    n_h = 84
    mids = np.array([init + pd.Timedelta(hours=k + 0.5) for k in range(n_h)], dtype="datetime64[ns]")
    snow = np.zeros((1, 9, n_h), "float32")
    snow[0, 4, :] = 0.1  # 0.1 in./h at the station's own cell
    snow[0, 0, :] = 0.2
    monkeypatch.setattr(nbmx, "_read", lambda i, s: (snow, mids))
    # local hours = hours since round-date midnight UTC (stand-in for the station's zone)
    monkeypatch.setattr(nbmx, "_local_hours",
                        lambda mid, s, d: ((pd.DatetimeIndex(mid) - pd.Timestamp(d)) / pd.Timedelta(hours=1))
                        .to_numpy()[None, :].astype("float32"))
    out = nbmx.features(init, st, date(2025, 12, 16)).set_index("horizon_h")
    assert out.loc[24, "nbmx_ctr"] == pytest.approx(2.4)
    assert out.loc[24, "nbmx_max"] == pytest.approx(4.8)
    assert out.loc[48, "nbmx_ctr"] == pytest.approx(4.8)
    assert out.loc[24, "nbmx_pre12"] == pytest.approx(1.2)
    assert out.loc[24, "nbmx_post6"] == pytest.approx(0.6)
    # the 72 h window ends at hour 84 after the 12z init: covered, the run is long enough
    assert out.loc[72, "nbmx_ctr"] == pytest.approx(7.2)


def test_alternative_row_scanning_is_undone():
    eccodes = pytest.importorskip("eccodes")
    ni, nj = 4, 3
    gid = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(gid, "Ni", ni)
        eccodes.codes_set(gid, "Nj", nj)
        eccodes.codes_set(gid, "alternativeRowScanning", 1)
        geo = np.arange(ni * nj, dtype=float).reshape(nj, ni)  # value = map position
        stored = geo.copy()
        stored[1::2] = stored[1::2, ::-1]  # how a boustrophedon file holds it
        eccodes.codes_set_values(gid, stored.reshape(-1))
        msg = eccodes.codes_get_message(gid)
    finally:
        eccodes.codes_release(gid)
    vals, _ = nbmx.decode(msg)
    np.testing.assert_allclose(vals, geo.reshape(-1))
