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
