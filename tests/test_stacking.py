from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from dpow import QCOLS, metrics
from dpow.models import gbm, learners, stacker


def test_board_is_mean_of_round_scores_and_skips_rounds_without_climatology_error():
    d = pd.DataFrame({
        "league": "stations", "station_id": ["A", "B", "A", "B", "A", "B"],
        "round_date": [date(2025, 1, 1)] * 2 + [date(2025, 1, 2)] * 2 + [date(2025, 1, 3)] * 2,
        "horizon_h": 24, "featured": 1,
        "truth_in": [10.0, 0.0, 0.0, 0.2, 0.0, 0.0],
        "climo_p50": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "f": [6.0, 0.0, 0.3, 0.3, 1.0, 1.0],
    })
    # round 1: 100*(1-4/10)=60; round 2: 100*(1-0.4/0.2)=-100; round 3 has no score
    assert metrics.round_scores(d, "f").round(6).tolist() == [60.0, -100.0]
    assert metrics.board(d, "f") == pytest.approx(-20.0)
    # pooled: errors 4+0.3+0.1+1+1=6.4 over climatology error 10.2
    assert metrics.pooled(d, "f") == pytest.approx(100 * (1 - 6.4 / 10.2))


def test_board_matches_powderbench_scoring():
    scoring = pytest.importorskip("powderbench.scoring")
    if not hasattr(scoring, "powder_score"):
        pytest.skip("powderbench.scoring.powder_score not exposed")
    assert scoring.powder_score(0.4, 1.0) == pytest.approx(60.0)
    assert scoring.powder_score(0.4, 0.0) is None


def _frame(n_days=240, seed=0):
    """Three seasons of two leagues; `q50` is a fair median, `mean` runs high,
    `p_wet` knows dry days and q50 drizzles a little on them."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, start in ((2023, date(2023, 11, 1)), (2024, date(2024, 11, 1)), (2025, date(2025, 11, 1))):
        for i in range(n_days // 2):
            d = start + timedelta(days=2 * i)
            wet_day = rng.random() < 0.35
            for lg in ("stations", "alps"):
                for sid in ("A", "B", "C", "D"):
                    sig = rng.gamma(1.5, 2.5) if wet_day else 0.0
                    truth = max(0.0, sig + rng.normal(0, 0.6)) if wet_day else 0.0
                    a = max(0.0, sig + rng.normal(0, 1.0)) if wet_day else abs(rng.normal(0, 0.15))
                    b = max(0.0, sig + rng.normal(0, 1.0)) if wet_day else abs(rng.normal(0, 0.15))
                    rows.append({"league": lg, "station_id": f"{lg}-{sid}", "round_date": d, "horizon_h": 24,
                                 "season": season, "fold": str(season), "featured": 1, "truth_in": truth,
                                 "climo_p50": 0.3, "q50": a, "mean": b,
                                 "q10": a * 0.5, "q25": a * 0.8, "q75": a * 1.2, "q90": a * 1.5,
                                 "p_wet": float(np.clip((0.9 if wet_day else 0.05) + rng.normal(0, 0.03), 0, 1))})
    return pd.DataFrame(rows)


def test_stacker_blends_independent_sources_and_zeroes_dry_days(monkeypatch):
    monkeypatch.setattr(stacker, "ROUND_K", 0.0)
    df = _frame()
    st = stacker.fit(df)
    c = st.cells[("stations", 24)]
    assert 0.2 <= c.weights.get("mean", 0) <= 0.8  # two equally noisy sources: both carry weight
    assert c.wet == "p_wet" and 0 < c.tau < 0.9
    p = stacker.predict(st, df)
    dry = (df["truth_in"] == 0).to_numpy()
    assert (p.loc[dry, "snowfall_in"] == 0).mean() > 0.95
    q = p[list(QCOLS)].to_numpy()
    assert (np.diff(q, axis=1) >= -1e-9).all() and (p["snowfall_in"] == p["p50"]).all()


def test_nested_evaluation_beats_the_anchor_on_both_rules():
    out, fitted = stacker.evaluate(_frame())
    assert out["stack"].notna().all() and len(fitted) == 3
    assert metrics.pooled(out, "stack") > metrics.pooled(out, "q50")
    assert metrics.board(out, "stack") > metrics.board(out, "q50") + 5


def test_predict_passes_the_anchor_through_for_unfitted_cells():
    df = _frame(40)
    p = stacker.predict(stacker.Stacker({}), df)
    assert np.allclose(p["snowfall_in"], df["q50"])
    assert (np.diff(p[list(QCOLS)].to_numpy(), axis=1) >= -1e-9).all()


def _table(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season, start in ((2023, date(2023, 11, 1)), (2024, date(2024, 11, 1)), (2025, date(2025, 11, 1))):
        for i in range(120):
            d = start + timedelta(days=i)
            for sid in ("A", "B", "C"):
                cons = rng.gamma(0.4, 3)
                rows.append({"league": "stations", "station_id": sid, "round_date": d, "horizon_h": 24,
                             "season": season, "featured": int(sid != "C"), "ref_snow": cons,
                             "gefs_snow_mean": cons * 1.2, "slr__m1": cons * 1.1, "cons_slr": cons, "nmod": 1,
                             "climo_p50": 0.2, "truth_in": max(0.0, 0.7 * cons + rng.normal(0, 0.3))})
    return pd.DataFrame(rows)


def test_learners_write_out_of_fold_columns_for_every_row(monkeypatch):
    monkeypatch.setattr(gbm, "MAX_ROUNDS", 60)
    df = _table()
    params = {**gbm.PARAMS, "min_child_samples": 5}
    q = learners.oof(df, "q50", params)
    assert len(q) == len(df) and set(q["fold"]) == {"2023", "2024", "2025"}
    h = learners.oof(df, "hurdle", params)
    assert {"p_wet", "p_ge1", "p_ge6"} <= set(h.columns) and h["p_wet"].between(0, 1).all()
    assert h["p_wet"].mean() > h["p_ge1"].mean() > h["p_ge6"].mean()
    joined = learners.join(df, [q, h, learners.oof(df, "tweedie", params)])
    assert len(joined) == len(df) and joined[["q50", "p_wet", "mean", "truth_in"]].notna().all().all()
    assert metrics.pooled(joined, "q50") > metrics.pooled(joined, "ref_snow")


def test_ensemble_only_learner_leaves_out_the_open_meteo_block():
    df = gbm.add_oof_ratio(_table())
    feats = learners.features_for(df, learners.LEARNERS["q50_ens"])
    assert "gefs_snow_mean" in feats and not {"slr__m1", "cons_slr", "nmod", "ref_snow"} & set(feats)
    assert {"slr__m1", "cons_slr"} <= set(learners.features_for(df, learners.LEARNERS["q50"]))
