from datetime import date

import numpy as np
import pandas as pd
import pytest

from dpow import dynom


def test_wide_spreads_steps_and_picks_3_hourly_temperatures():
    run = pd.Timestamp("2024-01-10 12:00")
    st = pd.DataFrame({"latitude": [40.0, 45.0], "longitude": [-111.0, 7.0]}, index=["a", "b"])
    rows = []
    for sid in ("a", "b"):
        for lead in range(1, 121):  # hourly to 120 h, then 3-hourly
            rows.append((sid, 0, run + pd.Timedelta(hours=lead), 1.0, 0.5, -2.0, 100.0 if lead <= 48 else -50.0))
        for lead in range(123, 157, 3):
            rows.append((sid, 0, run + pd.Timedelta(hours=lead), 3.0, 3.0, -4.0, 100.0))
    long = pd.DataFrame(rows, columns=["station_id", "member", "valid", "dt_h", "p", "t2m", "pfz"])
    df = dynom.wide(long, run, st)
    assert list(df["station_id"]) == ["a", "b"]
    assert np.allclose(df[[f"pr_{h:03d}" for h in range(1, 97)]].to_numpy(), 0.5)
    assert np.isnan(df["t2_00"]).all() and (df["t2_96"] == -2.0).all()
    assert np.isnan(df["td_48"]).all()
    assert np.allclose(df["sn_001"], 0.35) and np.allclose(df["sn_048"], 0.35)  # 0.5 mm all frozen x 0.7
    assert (df["sn_049"] == 0).all()  # -50: nothing frozen


def test_snowfall_reaches_the_daily_features():
    from dpow import ukmo

    run = pd.Timestamp("2024-01-10 12:00")
    st = pd.DataFrame({"latitude": [40.0], "longitude": [-111.0], "elevation": [2500.0]}, index=["a"])
    raw = pd.DataFrame({"station_id": ["a"], "run": [run]}
                       | {c: [0.5] for c in ukmo.columns() if c.startswith("pr_")}
                       | {c: [-5.0] for c in ukmo.columns() if c.startswith(("t2_", "td_"))}
                       | {c: [0.35] for c in ukmo.SNOW_COLS})
    h = ukmo.hourly(raw, st, pd.Series({"a": 0.0}), "gfs_dyn")
    assert np.allclose(h["snowfall"], 0.35)
    assert ukmo.hourly(raw.drop(columns=list(ukmo.SNOW_COLS)), st, pd.Series({"a": 0.0}))["snowfall"].isna().all()


def test_units_newest_month_first():
    u = dynom.units(["gfs", "hrrr"], pd.Timestamp("2024-03-10"))
    assert u[0][1] == pd.Period("2024-02", "M") and {u[0][0], u[1][0]} == {"gfs", "hrrr"}
    assert min(m for ds, m in u if ds == "gfs") == pd.Period("2021-05", "M")
    assert dynom.unit_name("gfs", pd.Period("2024-01", "M")) == "dynom__gfs__12z__2024-01.parquet"


def _long(run, hours=120, p=0.5, t=-5.0):
    rows = [(sid, 0, run + pd.Timedelta(hours=h), 1.0, p, t, 100.0) for sid in ("a",) for h in range(1, hours + 1)]
    return pd.DataFrame(rows, columns=["station_id", "member", "valid", "dt_h", "p", "t2m", "pfz"])


@pytest.mark.parametrize("missing", [False, True])
def test_round_windows_read_the_run_like_the_training_rows(monkeypatch, missing):
    import dpow.ens_features as ef

    st = pd.DataFrame({"latitude": [40.0], "longitude": [-111.0], "elevation_ft": [8000.0]}, index=["a"])
    calls = []

    def fake(ds, run, s):
        calls.append(run)
        if missing and run == pd.Timestamp("2024-01-09 12:00"):
            return None
        return dynom.wide(_long(run), run, s)

    monkeypatch.setattr(dynom, "harvest_run", fake)
    monkeypatch.setattr(dynom, "lapse_delta", lambda: pd.Series(dtype=float))
    monkeypatch.setattr(ef, "init_for_round", lambda rd, lg, ds: pd.Timestamp("2024-01-09 18:00"))
    w = dynom.round_windows("gfs", date(2024, 1, 10), "stations", st)
    assert (w["round_date"] == date(2024, 1, 10)).all()
    if missing:  # the run before: lead days shifted, and its 96 h stop short of the 72 h window
        assert calls == [pd.Timestamp("2024-01-09 12:00"), pd.Timestamp("2024-01-08 12:00")]
        assert sorted(w["horizon_h"]) == [24, 48]
    else:
        assert calls == [pd.Timestamp("2024-01-09 12:00")] and sorted(w["horizon_h"]) == [24, 48, 72]
    w24 = w[w["horizon_h"] == 24].iloc[0]
    assert np.isclose(w24["prcp__gfs_dyn"], 0.5 * 24 / 25.4, atol=1e-3)
    assert w24["snow__gfs_dyn"] > 0


def test_fill_takes_only_open_meteos_gaps_and_rebuilds_their_summaries():
    from dpow import features
    from dpow.dataset import add_ref

    d = date(2024, 1, 10)
    keys = {"station_id": ["a", "b"], "round_date": [d, d], "horizon_h": [24, 24]}
    df = pd.DataFrame(keys | {"slr__gfs_seamless": [2.0, np.nan], "prcp__gfs_seamless": [0.2, np.nan],
                              "slr__icon_seamless": [4.0, np.nan], "gefs_snow_q50": [1.0, 1.0]})
    features.summarize(df)
    add_ref(df)
    w = pd.DataFrame(keys | {"slr__gfs_dyn": [9.0, 3.0], "prcp__gfs_dyn": [0.9, 0.3]})
    out = dynom.fill(df, {"gfs": w})
    assert list(out["slr__gfs_seamless"]) == [2.0, 3.0] and "slr__gfs_dyn" not in out
    assert list(out["cons_slr"]) == [3.0, 3.0] and list(out["nmod"]) == [2, 1]
    assert out.loc[1, "ref_snow"] == 2.0
    # a live round without the Open-Meteo block: the twin columns are made
    bare = dynom.fill(df[["station_id", "round_date", "horizon_h", "gefs_snow_q50"]], {"gfs": w})
    assert list(bare["slr__gfs_seamless"]) == [9.0, 3.0] and list(bare["nmod"]) == [1, 1]
