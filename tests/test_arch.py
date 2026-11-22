from datetime import date

import pandas as pd

from dpow import arch


def test_units_put_the_new_ecmwf_months_first_and_skip_what_the_sibling_has(monkeypatch):
    monkeypatch.setattr(arch, "_all_featured", lambda lg: lg == "era5")
    u = arch.units(["stations", "era5"], date(2025, 2, 10))
    assert u[0][0] == "ecens" and u[0][2] < arch.SIBLING_ECENS_FROM
    # a league whose stations are all featured only gets the older ECMWF ENS months
    assert all(ds == "ecens" and m < arch.SIBLING_ECENS_FROM for ds, lg, m in u if lg == "era5")
    assert ("gefs", "stations", pd.Period("2024-12", "M")) in u
    assert not any(m.month in (6, 7, 8, 9) for ds, lg, m in u if lg == "stations")
    assert arch.unit_name("gfs", "stations", pd.Period("2024-12", "M")) == "ens__gfsf__stations-all__2024-12.parquet"
    assert arch.release(pd.Period("2024-12", "M")) == "arch-2024"


def test_round_dates_are_the_siblings(monkeypatch):
    seen = []
    monkeypatch.setattr("dpow.contract.stations", lambda lg: pd.DataFrame(index=["a"]))
    monkeypatch.setattr("dpow.ens_features.init_for_round", lambda rd, lg, ds: rd)
    monkeypatch.setattr("dpow.ens_features.features_for_init",
                        lambda ds, init, st, lg, round_date: seen.append(round_date) or pd.DataFrame({"x": [1]}))
    arch.harvest_unit("gefs", "stations", pd.Period("2021-01", "M"), deadline=float("inf"))
    assert [d.day for d in seen] == [*range(2, 31, 2), 1] and seen[-1] == date(2021, 2, 1)
    seen.clear()
    arch.harvest_unit("gefs", "stations", pd.Period("2021-02", "M"), deadline=float("inf"))
    assert [d.day for d in seen] == list(range(2, 29, 2))
    assert arch.harvest_unit("gefs", "stations", pd.Period("2021-02", "M"), deadline=0) is None
