from datetime import date, timedelta

import numpy as np
import pandas as pd

from dpow import QCOLS
from dpow.models import gbm


def _table(n_days=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        d = date(2024, 11, 1) + timedelta(days=i)
        for sid in ("A", "B", "C"):
            for h in (24, 48, 72):
                cons = rng.gamma(0.4, 3) * h / 24
                rows.append({"station_id": sid, "round_date": d, "horizon_h": h, "ref_snow": cons,
                             "slr__m1": cons * 1.1, "slr__m2": cons * 0.9, "nmod": 2,
                             "truth_in": max(0.0, 0.7 * cons + rng.normal(0, 0.5)),
                             "season": 2024 if d < date(2025, 8, 1) else 2025, "aug": 0})
    return pd.DataFrame(rows)


def test_fit_predict_is_calibrated_and_monotone(monkeypatch):
    monkeypatch.setattr(gbm, "MAX_ROUNDS", 200)
    df = _table()
    b = gbm.fit(df)
    assert b.metrics["mae"] > 0 and b.metrics["holdout_season"] == "2025"
    X = df[df.season == 2025].drop(columns=["truth_in"])
    out = gbm.predict(b, X)
    q = out[list(QCOLS)].to_numpy()
    assert (np.diff(q, axis=1) >= -1e-9).all()
    assert (out["snowfall_in"] == out["p50"]).all()
    # learned the 0.7 ratio: the median tracks 0.7 * consensus, not 1.0
    big = X["ref_snow"].to_numpy() > 4
    ratio = out.loc[big, "p50"].sum() / X.loc[big, "ref_snow"].sum()
    assert 0.55 < ratio < 0.85


def test_holdout_skips_a_season_that_has_barely_started(monkeypatch):
    monkeypatch.setattr(gbm, "MAX_ROUNDS", 100)
    df = _table(n_days=283)  # 273 days of season 2024, then 10 days of 2025 (3.5% of rows)
    b = gbm.fit(df)
    assert b.metrics["holdout_season"] == "2024-tail"  # not the 10-day season, and not all of 2024 either
    assert b.metrics["n_train"] > 0.6 * len(df)
    assert not any(hasattr(m, "train_set") for m in b.quantile_models.values())  # boosters don't pin their data


def _seasons(n=3):
    """Three full seasons of the synthetic table, with what experiments need."""
    parts = []
    for k in range(n):
        t = _table(n_days=120, seed=k)
        t["round_date"] = [d.replace(year=d.year + k) for d in t["round_date"]]
        t["season"] = 2024 + k
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    df["league"], df["featured"], df["climo_p50"] = "stations", (df["station_id"] == "A").astype(int), 0.3
    return df


def test_experiments_score_variants_on_the_same_folds(monkeypatch):
    from dpow.models import experiments

    monkeypatch.setattr(gbm, "MAX_ROUNDS", 60)
    df = _seasons()
    md = experiments.run(df, ["w_feat4", "resid", "seedbag", "moddrop", "f_spread"], gbm.PARAMS)
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "variant" not in ln]
    assert [r.split("|")[1].strip() for r in rows] == ["base", "w_feat4", "resid", "seedbag", "moddrop", "f_spread"]
    assert all(r.split("|")[2].strip() not in ("", "None") for r in rows)  # every variant got a CV score
    assert not any(c.startswith("_") for c in gbm.feature_list(df))  # weight columns never become features


def test_experimental_columns_stay_out_until_asked_for():
    df = _seasons().assign(q25_slr=1.0, cons_tmin=-3.0)
    assert "q25_slr" not in gbm.feature_list(df) and "cons_tmin" not in gbm.feature_list(df)
    assert "q25_slr" in gbm.feature_list(df, ("q25_slr",))


def test_featured_rows_weigh_more_in_training(monkeypatch):
    monkeypatch.setattr(gbm, "MAX_ROUNDS", 60)
    df = _seasons()
    w = gbm.row_weight(df)
    assert set(w) == {1.0, gbm.FEATURED_WEIGHT} and (w > 1).sum() == df["featured"].sum()
    b = gbm.fit(df)
    assert len(gbm.predict(b, df.drop(columns=["truth_in"]))) == len(df)


def test_snowfall_family_follows_the_override():
    """A row with models but no snowfall (a direct-source fill) must not switch
    the models' own snowfall off when the variant keeps the base's decision."""
    df = _seasons().assign(nmod=1, nsnow=0, cons_snow=1.0, snow__x=1.0)
    assert not gbm.snow_ready(df)
    assert "cons_snow" not in gbm.feature_list(df)
    assert {"cons_snow", "snow__x"} <= set(gbm.feature_list(df, snow=True))
    assert "cons_snow" not in gbm.feature_list(df.assign(nsnow=3), snow=False)


def test_chained_variants_build_on_each_others_tables():
    """`_then`: each step gets the table the previous one returned, and a later
    step's spec keys win (gfs_allst25 = extra stations, then the GFS fill on all rows)."""
    from dpow.models import experiments

    seen = []

    def add_rows(df):
        seen.append(len(df))
        return {"df": pd.concat([df, df.head(2)], ignore_index=True), "snow": False}

    def fill(df):
        seen.append(len(df))
        return {"df": df.assign(filled=1.0), "snow": True}

    df = _seasons()
    out = experiments._then(add_rows, fill)(df)
    assert seen == [len(df), len(df) + 2]
    assert len(out["df"]) == len(df) + 2 and out["df"]["filled"].eq(1.0).all() and out["snow"] is True
    assert {"gfs_ecens_old", "gfs_allst25", "gfs_allst50"} <= set(experiments.VARIANTS)


def test_old_ecmwf_ens_fills_float32_columns_from_float64_units(monkeypatch):
    """ecens_old: the archive's float64 values go into the table's float32
    columns only where the table had none (pandas refuses the plain setitem)."""
    from dpow import arch
    from dpow.models import experiments

    df = _seasons()
    df["ecens_snow_q50"] = np.where(np.arange(len(df)) % 2 == 0, np.nan, 5.0).astype("float32")
    keys = experiments.ARCH_KEYS
    e = df[keys].copy()
    e["ecens_snow_q50"] = np.linspace(0.001, 0.9, len(e))  # float64, not exact in float32
    monkeypatch.setattr(arch, "pull", lambda *a, **k: None)
    monkeypatch.setattr(arch, "load", lambda lg, ds=None: e)
    monkeypatch.setattr(gbm, "add_oof_ratio", lambda d: d)
    out = experiments._ecens_old(df)["df"]
    assert out["ecens_snow_q50"].dtype == np.float32 and out["ecens_snow_q50"].notna().all()
    odd = np.arange(len(df)) % 2 == 1
    assert (out.loc[odd, "ecens_snow_q50"] == 5.0).all()  # existing values kept
    assert np.allclose(out.loc[~odd, "ecens_snow_q50"], e.loc[~odd, "ecens_snow_q50"], atol=1e-6)
    assert df["ecens_snow_q50"].isna().sum() == (~odd).sum()  # the base table is untouched
