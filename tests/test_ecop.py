import json

import numpy as np
import pandas as pd

from dpow import ecop


def test_file_names_and_layouts():
    r = pd.Timestamp("2025-01-15 12:00")
    assert ecop.file_url(r, "ifs/0p25/oper", 24) == (
        f"{ecop.BUCKET}/20250115/12z/ifs/0p25/oper/20250115120000-24h-oper-fc.grib2")
    idx = ecop.file_url(r, "0p4-beta/oper", 0, "index")
    assert idx.endswith("/20250115/12z/0p4-beta/oper/20250115120000-0h-oper-fc.index")
    assert ecop.unit_name(12, pd.Period("2025-01", "M")) == "ecop__12z__2025-01.parquet"


def test_messages_keeps_surface_fields_only():
    idx = "\n".join(json.dumps(d) for d in [
        {"param": "tp", "levtype": "sfc", "_offset": 10, "_length": 5},
        {"param": "t", "levtype": "pl", "levelist": "850", "_offset": 15, "_length": 5},
        {"param": "2t", "levtype": "sfc", "_offset": 20, "_length": 7},
        {"param": "msl", "levtype": "sfc", "_offset": 27, "_length": 3},
    ])
    assert ecop.messages(idx) == {"tp": (10, 5), "2t": (20, 7)}


def test_cell_index_on_the_open_data_grid():
    # 0.25 deg: lat 90 -> -90 (jScansPositively 0), lon -180 -> 179.75
    g = (1440, 721, 90.0, -180.0, 0.25, 0.25, 0)
    lat, lon = np.array([90.0, 40.6, -33.0, 0.0]), np.array([-180.0, -111.6, 179.9, 0.1])
    flat = ecop.cell_index(g, lat, lon)
    j, i = np.divmod(flat, 1440)
    assert list(j) == [0, 198, 492, 360]
    assert list(i) == [0, 274, 0, 720]  # 179.9 wraps to -180


def test_units_put_northern_winter_first():
    u = ecop.units(pd.Timestamp("2024-06-10"), hours=(12,))
    assert u[0] == (12, pd.Period("2023-01", "M")) and (12, pd.Period("2024-05", "M")) in u
    first_rest = next(k for k, (h, m) in enumerate(u) if m.month not in (11, 12, 1, 2, 3, 4))
    assert all(m.month in (11, 12, 1, 2, 3, 4) for h, m in u[:first_rest])
    runs = ecop.runs_in(12, pd.Period("2023-01", "M"), pd.Timestamp("2024-01-01"))
    assert runs[0] == pd.Timestamp("2023-01-18 12:00")


def test_harvest_run_spreads_accumulated_precipitation(monkeypatch):
    st = pd.DataFrame({"latitude": [40.0], "longitude": [-111.0]}, index=["a"])
    monkeypatch.setattr(ecop, "layout_for", lambda r: "ifs/0p25/oper")

    def step(run, layout, s, lat, lon):  # 0.6 mm per 3 h, t2 0 C, no dew point
        return {"tp": np.array([s * 0.0002], np.float32), "t2": np.array([273.15], np.float32)}

    monkeypatch.setattr(ecop, "_read_step", step)

    class Pool:
        def map(self, f, xs):
            return [f(x) for x in xs]

    df = ecop.harvest_run(pd.Timestamp("2025-01-15 12:00"), st, Pool())
    assert np.allclose(df[[f"pr_{h:03d}" for h in range(1, 97)]].to_numpy(), 0.2)
    assert df["t2_48"].iloc[0] == 0.0 and np.isnan(df["td_48"].iloc[0])
