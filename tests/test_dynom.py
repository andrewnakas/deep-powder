import numpy as np
import pandas as pd

from dpow import dynom


def test_wide_spreads_steps_and_picks_3_hourly_temperatures():
    run = pd.Timestamp("2024-01-10 12:00")
    st = pd.DataFrame({"latitude": [40.0, 45.0], "longitude": [-111.0, 7.0]}, index=["a", "b"])
    rows = []
    for sid in ("a", "b"):
        for lead in range(1, 121):  # hourly to 120 h, then 3-hourly
            rows.append((sid, 0, run + pd.Timedelta(hours=lead), 1.0, 0.5, -2.0))
        for lead in range(123, 157, 3):
            rows.append((sid, 0, run + pd.Timedelta(hours=lead), 3.0, 3.0, -4.0))
    long = pd.DataFrame(rows, columns=["station_id", "member", "valid", "dt_h", "p", "t2m"])
    df = dynom.wide(long, run, st)
    assert list(df["station_id"]) == ["a", "b"]
    assert np.allclose(df[[f"pr_{h:03d}" for h in range(1, 97)]].to_numpy(), 0.5)
    assert np.isnan(df["t2_00"]).all() and (df["t2_96"] == -2.0).all()
    assert np.isnan(df["td_48"]).all()


def test_units_newest_month_first():
    u = dynom.units(["gfs", "hrrr"], pd.Timestamp("2024-03-10"))
    assert u[0][1] == pd.Period("2024-02", "M") and {u[0][0], u[1][0]} == {"gfs", "hrrr"}
    assert min(m for ds, m in u if ds == "gfs") == pd.Period("2021-05", "M")
    assert dynom.unit_name("gfs", pd.Period("2024-01", "M")) == "dynom__gfs__12z__2024-01.parquet"
