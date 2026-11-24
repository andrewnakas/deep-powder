import numpy as np
import pandas as pd

from dpow import ukmo


def test_file_names_follow_the_bucket():
    run = pd.Timestamp("2025-12-01 00:00")
    assert ukmo.file_url(run, 12, "pr1").endswith(
        "global-deterministic-10km/20251201T0000Z/20251201T1200Z-PT0012H00M-precipitation_accumulation-PT01H.nc")
    assert ukmo.file_url(run, 54, "td").endswith(
        "20251203T0600Z-PT0054H00M-temperature_of_dew_point_at_screen_level.nc")


def test_tasks_cover_every_hour_of_precipitation_once():
    t = ukmo.tasks(pd.Timestamp("2025-12-01 12:00"))
    hours = [h for k, _, lead in t if k == "pr1" for h in [lead]]
    hours += [h for k, _, lead in t if k == "pr3" for h in (lead - 2, lead - 1, lead)]
    assert sorted(hours) == list(range(1, ukmo.MAX_LEAD + 1))
    assert len(ukmo.columns()) == ukmo.MAX_LEAD + 2 * len(ukmo.T_LEADS)


def test_merge_ranges():
    assert ukmo.merge_ranges([(100, 10), (0, 50), (200_000, 5)], gap=100) == [(0, 110), (200_000, 5)]
    assert ukmo.merge_ranges([(0, 50), (10, 20)]) == [(0, 50)]


def test_units_are_complete_months_oldest_first():
    u = ukmo.units(pd.Timestamp("2025-02-10"))
    assert u[0] == (12, pd.Period("2024-11", "M")) and (0, pd.Period("2025-01", "M")) in u
    assert all(m <= pd.Period("2025-01", "M") for _, m in u)
    runs = ukmo.runs_in(12, pd.Period("2024-11", "M"), pd.Timestamp("2025-02-10"))
    assert runs[0] == pd.Timestamp("2024-11-21 12:00") and len(runs) == 10


def test_hourly_puts_local_days_on_the_previous_runs_leads():
    st = pd.DataFrame({"latitude": [40.6], "longitude": [-111.6], "elevation_ft": [8000.0]},
                      index=pd.Index(["a"], name="station_id"))
    row = {"station_id": "a", "run": pd.Timestamp("2025-12-01 12:00")}
    row |= {f"pr_{h:03d}": 1.0 for h in range(1, ukmo.MAX_LEAD + 1)}
    row |= {f"t2_{h:02d}": float(h) for h in ukmo.T_LEADS} | {f"td_{h:02d}": 0.0 for h in ukmo.T_LEADS}
    h = ukmo.hourly(pd.DataFrame([row]), st, pd.Series({"a": 100.0}))
    assert len(h) == ukmo.MAX_LEAD and (h["model"] == ukmo.MODEL).all()
    # Utah is UTC-7: the hour ending 07Z Dec 2 (local midnight) closes local Dec 1, lead 0
    first_d = h[h["lead"] == 1].iloc[0]
    assert first_d["time"] == pd.Timestamp("2025-12-02 01:00")
    assert (h.groupby("lead").size().loc[1:3] == 24).all()
    # linear between the 3-hourly values, then 0.65 C colder for 100 m above the cell's terrain
    assert abs(h.iloc[0]["temperature_2m"] - (1.0 - 0.65)) < 1e-4


def test_hourly_drops_stations_no_longer_registered():
    st = pd.DataFrame({"latitude": [40.6], "longitude": [-111.6], "elevation_ft": [8000.0]},
                      index=pd.Index(["a"], name="station_id"))
    row = {"station_id": "a", "run": pd.Timestamp("2025-12-01 12:00")}
    row |= {f"pr_{h:03d}": 1.0 for h in range(1, ukmo.MAX_LEAD + 1)}
    row |= {f"t2_{h:02d}": 0.0 for h in ukmo.T_LEADS} | {f"td_{h:02d}": 0.0 for h in ukmo.T_LEADS}
    gone = row | {"station_id": "6BE0:CH:SLF"}  # harvested, then dropped from the registry
    h = ukmo.hourly(pd.DataFrame([row, gone]), st, pd.Series({"a": 0.0}))
    assert len(h) == ukmo.MAX_LEAD and set(h["station_id"]) == {"a"}

def test_windows_from_one_unit_at_a_time_matches_all_at_once(tmp_path, monkeypatch):
    from dpow import features, ukmo

    st = pd.DataFrame({"latitude": [40.6], "longitude": [-111.6], "elevation_ft": [9000.0]}, index=["s1"])
    monkeypatch.setattr(ukmo, "stations", lambda: st)
    monkeypatch.setattr(features, "terrain", lambda: pd.DataFrame({"elev_delta_14km": [0.0]}, index=["s1"]))
    rng = np.random.default_rng(1)
    files = []
    for month, days in (("2025-01", range(26, 32)), ("2025-02", range(1, 6))):
        rows = []
        for d in days:
            run = pd.Timestamp(f"{month}-{d:02d} 12:00")
            r = {"station_id": "s1", "run": run}
            r |= {f"pr_{h:03d}": float(rng.gamma(0.5, 1.0)) for h in range(1, ukmo.MAX_LEAD + 1)}
            r |= {f"t2_{h:02d}": float(-5 + rng.normal()) for h in ukmo.T_LEADS}
            r |= {f"td_{h:02d}": np.nan for h in ukmo.T_LEADS}
            rows.append(r)
        p = tmp_path / f"ukmo__12z__{month}.parquet"
        pd.DataFrame(rows).to_parquet(p)
        files.append(p)
    split = ukmo.windows_from(files, "m")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    one = tmp_path / "one" / "ukmo__12z__all.parquet"
    one.parent.mkdir()
    raw.to_parquet(one)
    whole = ukmo.windows_from([one], "m")
    keys = ["station_id", "round_date", "horizon_h"]
    a, b = split.sort_values(keys).reset_index(drop=True), whole.sort_values(keys).reset_index(drop=True)
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False)
