"""dpow — the deep-powder command line (what the workflows run)."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from . import ROOT, use_pb_data

use_pb_data()
app = typer.Typer(add_completion=False, no_args_is_help=True)
# Submit only leagues with a trained bundle: a raw-ensemble tier 0 would just
# drag the season score down while the archive is still being harvested.
REQUIRE_BUNDLE = os.environ.get("DPOW_REQUIRE_BUNDLE", "1") == "1"
LeaguesOpt = typer.Option(None, "--league", "-l", help="Limit to these leagues (repeatable)")


def _summary(md: str) -> None:
    print(md)
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a") as f:
            f.write(md + "\n")


@app.command()
def rounds():
    """List the open rounds and their cutoffs."""
    from .contract import open_rounds

    now = datetime.now(timezone.utc)
    for r in open_rounds(now):
        print(f"{r.key:24s} cutoff {r.cutoff:%Y-%m-%d %H:%MZ} ({r.hours_left(now):5.1f} h)  "
              f"{len(r.stations)} stations, {len(r.featured)} featured")


@app.command()
def due(league: list[str] | None = LeaguesOpt):
    """What a forecast run would do now."""
    from .schedule import due as _due
    from .schedule import latest_cycle
    from .submit import upstream_file

    # with a token, also catch submissions that never merged
    on_up = (lambda r: upstream_file(r) is not None) if os.environ.get("POWDERBENCH_TOKEN") else None
    acts = _due(leagues=league, on_upstream=on_up)
    print(f"cycle {latest_cycle(datetime.now(timezone.utc))}")
    for a in acts:
        print(f"{a.reason:9s} {a.round.key} (cutoff {a.round.cutoff:%m-%d %H:%MZ})")
    if not acts:
        print("nothing due")


@app.command()
def run(
    league: list[str] | None = LeaguesOpt,
    dry_run: bool = typer.Option(False, help="Build and validate, don't submit"),
    shadow: bool = typer.Option(False, help="Build, validate and save to history/, but don't submit"),
    force: bool = typer.Option(False, help="Treat every open round as due (still deadline-guarded)"),
    network: bool = typer.Option(True, help="Also forecast non-featured network stations"),
    budget: float = typer.Option(8000, help="Open-Meteo weighted-call budget for this run"),
    wait: int = typer.Option(240, help="Seconds to wait for the auto-merge per submission"),
):
    """Forecast and submit every due round."""
    from powderbench.validate import validate_submission

    from . import predict, schedule, submit

    on_up = None if dry_run or shadow else (lambda r: submit.upstream_file(r) is not None)
    acts = schedule.due(leagues=league, force=force, on_upstream=on_up)
    if not acts:
        print("nothing due")
        return
    failures = 0
    for a in acts:
        r = a.round
        try:
            if REQUIRE_BUNDLE and not dry_run and not predict.have_model(r.league):
                _summary(f"**{r.key}** skipped: no trained model yet (raw ensembles aren't worth submitting)")
                continue
            res = predict.build(r, network=network, budget=budget)
            out = ROOT / "out" / r.league / f"{r.round_date}.csv"
            predict.write(res, out)
            v = validate_submission(out, league=r.league, round_stations=list(r.stations))
            if not v.ok:
                raise RuntimeError("validation failed:\n" + v.summary())
            line = (f"**{r.key}** ({a.reason}) · cutoff {r.cutoff:%m-%d %H:%MZ} · {res.info['rows']} rows · "
                    f"featured coverage {v.coverage:.0%} · tiers {res.info['tiers']} · "
                    f"runs {res.info.get('gefs_init', '-')}/{res.info.get('ecens_init', '-')} · "
                    f"multi-model block {res.info.get('det_block', '-')}")
            errs = {k: v for k, v in res.info.items() if k.endswith("_error")}
            if errs:
                line += f" · {errs}"
            if dry_run:
                _summary(line + " · DRY RUN")
                continue
            hist = schedule.HISTORY / r.league / f"{r.round_date}.csv"
            hist.parent.mkdir(parents=True, exist_ok=True)
            res.raw.round(3).to_csv(hist, index=False)
            predict.save_features(res, hist.with_suffix(".features.parquet"), reason=a.reason)
            if shadow:
                _summary(line + " · SHADOW (saved, not submitted)")
                continue
            st = schedule.load_state(r)
            cyc = schedule.latest_cycle(datetime.now(timezone.utc))
            note = (f"Model: {res.info['bundle'] or 'tier-0 ensemble'} · GEFS {res.info.get('gefs_init')}Z, "
                    f"ECMWF ENS {res.info.get('ecens_init')}Z · cycle {cyc} · rows {res.info['rows']}")
            sub = submit.submit(r, out, note=note)
            st.update(
                submissions=st.get("submissions", 0) + 1,
                updates=st.get("updates", 0) + (a.reason == "update"),
                final=st.get("final", False) or a.reason == "final",
                cycle=cyc, submitted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                pr=sub.pr_url, bundle=res.info["bundle"], tiers=res.info["tiers"],
            )
            schedule.save_state(r, st)
            merged = submit.wait_merged(r, out.read_text(), timeout_s=wait) if wait else None
            st["merged"] = merged
            schedule.save_state(r, st)
            _summary(f"{line} · [PR]({sub.pr_url}) · merged: {merged}")
        except submit.TooLate as e:
            _summary(f"**{r.key}** skipped: {e}")
        except Exception as e:  # one league failing must not stop the others
            failures += 1
            _summary(f"**{r.key}** FAILED: {type(e).__name__}: {e}")
    if failures:
        raise typer.Exit(1)


def _build(bundle: str):
    """The training table of a bundle, built from the sibling entries' stores."""
    import pandas as pd

    from . import NH_LEAGUES, store
    from .dataset import training_table
    from .ens_store import ENS_STORE

    store.pull_ens(ENS_STORE)
    store.pull("store", store.STORE, repo=store.DET_REPO)  # deterministic archive features (public)
    leagues = NH_LEAGUES if bundle == "nh" else (bundle,)
    tables = [t for t in (training_table(lg) for lg in leagues) if len(t)]
    if not tables:
        return None
    df = pd.concat(tables, ignore_index=True)
    del tables
    if bundle == "nh":
        from . import nbhd, wndx

        df = wndx.join(nbhd.join(df))
    # half the memory: LightGBM bins the features anyway, so float32 loses nothing
    f64 = [c for c in df.select_dtypes("float64").columns if c != "truth_in"]
    df[f64] = df[f64].astype("float32")
    return df


def _table(bundle: str, columns: list[str] | None = None, rebuild: bool = False):
    """The table from the `table` release (one parquet per league and season),
    or built on the spot when the release has none for this bundle."""
    import pandas as pd

    from . import store

    if not rebuild:
        store.pull("table", store.TABLE, f"{bundle}__*.parquet")
        files = sorted(store.TABLE.glob(f"{bundle}__*.parquet"))
        if files:
            df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
            df["round_date"] = pd.to_datetime(df["round_date"]).dt.date
            return df
    df = _build(bundle)
    return df if df is None or columns is None else df[[c for c in columns if c in df]]


@app.command()
def table(bundle: str = typer.Option("nh", help="nh (stations+observers+alps), era5, resorts or resorts-na")):
    """Build the training table and publish it to the `table` release, so no
    training job (here or off Actions) has to rebuild it."""
    from . import store

    df = _build(bundle)
    if df is None:
        _summary(f"table {bundle}: no data yet")
        return
    store.TABLE.mkdir(parents=True, exist_ok=True)
    files = []
    for (lg, season), g in df.groupby(["league", "season"]):
        f = store.TABLE / f"{bundle}__{lg}__{int(season)}.parquet"
        g.to_parquet(f, index=False, compression="zstd")
        files.append(f)
    stale = store.remote_assets("table") - {f.name for f in files}
    stale = {n for n in stale if n.startswith(f"{bundle}__")}
    store.push("table", files)
    cover = df.groupby("league").size().to_dict()
    _summary(f"table {bundle}: {len(df)} rows, {df.shape[1]} columns, seasons "
             f"{sorted(int(s) for s in df.season.unique())}, by league {cover}, {len(files)} files, "
             f"{sum(f.stat().st_size for f in files) / 1e6:.0f} MB, peak memory {_peak_gb():.1f} GB"
             + (f"; stale assets left in the release: {sorted(stale)}" if stale else ""))


@app.command()
def oof(
    learner: str = typer.Option(..., help="A name from models/learners.py"),
    bundle: str = typer.Option("nh"),
    final: bool = typer.Option(True, help="Also fit on all seasons and publish the model"),
):
    """Out-of-fold predictions of one level-1 learner, published to the `oof`
    release; then the same learner fitted on all seasons, to `models`."""
    import pickle

    from . import WORK, store
    from .models import gbm, learners

    df = _table(bundle)
    if df is None:
        _summary(f"oof {bundle}: no table")
        return
    store.pull("models", store.MODELS, f"{bundle}.json")
    params = gbm.REFERENCE
    mj = store.MODELS / f"{bundle}.json"
    if mj.exists():  # the settings the season CV picked for the reference model
        params = json.loads(mj.read_text()).get("params") or params
    info: dict = {}
    out = learners.oof(df, learner, params, info=info)
    d = WORK / "oof"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{bundle}__{learner}.parquet"
    out.to_parquet(f, index=False, compression="zstd")
    fj = d / f"{bundle}__{learner}.json"
    fj.write_text(json.dumps({**info, "rows": len(df), "trained_through": str(max(df["round_date"]))}, indent=1))
    store.push("oof", [f, fj])
    _summary(f"oof {bundle} {learner}: {len(out)} rows, columns "
             f"{[c for c in out.columns if c not in (*learners.KEYS, 'fold')]}, rounds {info.get('rounds')}, "
             f"peak memory {_peak_gb():.1f} GB")
    if final:
        del out
        fm = learners.fit_final(df, learner, info)
        store.MODELS.mkdir(parents=True, exist_ok=True)
        fp = store.MODELS / f"{bundle}__{learner}.pkl"
        fp.write_bytes(pickle.dumps(fm))
        store.push("models", [fp])
        _summary(f"final {bundle} {learner}: trained through {fm.trained_through}, {fp.stat().st_size / 1e6:.0f} MB, "
                 f"peak memory {_peak_gb():.1f} GB")


@app.command()
def stack(bundle: str = typer.Option("nh")):
    """Fit the level-2 stacker on the published out-of-fold predictions and
    report every learner and the stack on the same rows, both board rules."""
    import pandas as pd

    from . import QCOLS, WORK, metrics, store
    from .models import learners, stacker

    d = WORK / "oof"
    store.pull("oof", d, f"{bundle}__*.parquet")
    frames = {f.stem.split("__", 1)[1]: pd.read_parquet(f) for f in sorted(d.glob(f"{bundle}__*.parquet"))}
    frames = {k: v for k, v in frames.items() if k in learners.LEARNERS}  # not our own earlier `stack` output
    if stacker.ANCHOR not in frames:
        _summary(f"stack {bundle}: no out-of-fold `{stacker.ANCHOR}` yet")
        return
    meta = _table(bundle, columns=learners.META)
    order = [stacker.ANCHOR, *[k for k in frames if k != stacker.ANCHOR]]
    df = learners.join(meta, [frames[k] for k in order])
    df["l2"] = stacker.l2_oof(df)
    df, fitted = stacker.evaluate(df)
    points = [c for c in ("ref_snow", *stacker.POINT_SOURCES) if c in df and c != "cons_snow"]
    points = list(dict.fromkeys(points))
    scored = df.dropna(subset=["stack"])
    _summary(f"## stack {bundle}: learners {order}\n")
    _summary(metrics.table(scored, [*points, "stack"], base="stack"))
    if "cons_snow" in scored and scored["cons_snow"].notna().sum() > 1000:
        _summary("\nRows that have the models' own snowfall consensus (what baseline-consensus is):\n")
        _summary(metrics.table(scored, ["cons_snow", stacker.ANCHOR, "stack"], base="stack"))
    rows = metrics.scored_rows(scored)
    pin = {"stack": metrics.pinball(rows, tuple(f"stack_{q}" for q in QCOLS))}
    if all(c in rows for c in ("q10", "q25", "q75", "q90")):
        pin["level-1 quantiles"] = metrics.pinball(rows, ("q10", "q25", "q50", "q75", "q90"))
    _summary("\npinball: " + ", ".join(f"{k} {v:.4f}" for k, v in pin.items() if v is not None))
    for c in ("p_ge6", "climo_p6"):
        b = metrics.brier6(rows, c)
        if b is not None:
            _summary(f"brier 6 in. ({c}): {b:.4f}")
    if fitted:
        _summary("\nStacker fitted on all but the last fold:\n\n" + fitted[-1].describe())
    # the stacker that goes live is fitted on every fold
    import pickle

    live = stacker.fit(df.dropna(subset=["fold"]))
    store.MODELS.mkdir(parents=True, exist_ok=True)
    sp = store.MODELS / f"{bundle}__stacker.pkl"
    sp.write_bytes(pickle.dumps({"stacker": live, "learners": order, "l2": stacker.l2_fit(df.dropna(subset=["fold"]))}))
    store.push("models", [sp])
    _summary("\nStacker fitted on all folds (the one used live):\n\n" + live.describe())
    f = d / f"{bundle}__stack.parquet"
    keep = [*learners.KEYS, "fold", "stack", *(f"stack_{q}" for q in QCOLS)]
    scored[keep].to_parquet(f, index=False, compression="zstd")
    store.push("oof", [f])


@app.command()
def train(
    bundle: str = typer.Option("nh", help="nh (stations+observers+alps), era5, resorts or resorts-na"),
    promote: bool = typer.Option(True, help="Upload as champion if it beats the current one"),
):
    """Train the reference bundle (one LightGBM quantile model, as in
    open-powder-ensemble), score it on the held-out season, promote if better."""
    from . import store
    from .models import gbm

    df = _table(bundle)
    if df is None:
        _summary(f"train {bundle}: no training data yet")
        return
    store.pull("models", store.MODELS)
    if bundle == "nh":  # never silently train without the snowpack state again
        st = df[df["league"] == "stations"]
        cov = st["sp_depth"].notna().mean() if "sp_depth" in st and len(st) else 0.0
        _summary(f"snowpack features on {cov:.0%} of stations-league rows")
    path = store.MODELS / f"{bundle}.pkl"
    old = gbm.Bundle.load(path) if path.exists() else None
    b = gbm.fit(df, champion_params=getattr(old, "params", None) if old else None)
    b.metrics["params"] = b.params
    _summary(f"train {bundle}: {len(df)} rows, seasons {sorted(int(s) for s in df.season.unique())}, "
             f"{len(b.features)} features, rounds {b.rounds}, peak memory {_peak_gb():.1f} GB")
    _summary("held-out: " + json.dumps({k: v for k, v in b.metrics.items() if k not in ("ref_snow_p99", "cv")}))
    if b.metrics.get("cv"):
        _summary("season CV (Powder Accuracy; raw ensemble+model consensus as the baseline): " + "; ".join(
            f"{r['label']} {r['powder_score']} vs raw {r['raw_consensus']} (folds {r['folds']})"
            for r in b.metrics["cv"]["all"]))
    cand = store.MODELS / f"{bundle}.candidate.pkl"
    b.save(cand)
    better = _better(b.metrics, old.metrics if old else None)
    _summary(f"champion: {'none' if old is None else old.metrics.get('powder_score')} -> "
             f"candidate {b.metrics.get('powder_score')}: {'PROMOTE' if better else 'keep champion'}")
    if better:
        cand.replace(path)
        (store.MODELS / f"{bundle}.json").write_text(json.dumps(b.metrics, indent=1, default=str))
        if promote:
            store.push("models", [path, store.MODELS / f"{bundle}.json"])


@app.command()
def experiment(
    bundle: str = typer.Option("nh", help="nh (stations+observers+alps), era5, resorts or resorts-na"),
    variants: str = typer.Option("", help="Comma-separated names from models/experiments.py (empty: list them)"),
    params: str = typer.Option("", help="JSON of LightGBM settings (default: the bundle's, else gbm.REFERENCE)"),
):
    """Season CV of the reference settings against named variants, same folds
    and rows. Writes no model."""
    from . import store
    from .models import experiments, gbm

    names = [v.strip() for v in variants.split(",") if v.strip()]
    if not names:
        print("variants: base, " + ", ".join(experiments.VARIANTS))
        return
    df = _table(bundle)
    if df is None:
        _summary(f"experiment {bundle}: no training data yet")
        return
    store.pull("models", store.MODELS, f"{bundle}.json")
    mj = store.MODELS / f"{bundle}.json"
    use = json.loads(params) if params else (json.loads(mj.read_text()).get("params") if mj.exists() else None)
    _summary(experiments.run(df, [n for n in names if n != "base"], use or gbm.REFERENCE))
    _summary(f"peak memory {_peak_gb():.1f} GB")


def _peak_gb() -> float:
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kilobytes on Linux, bytes on macOS
    return peak / (1e9 if sys.platform == "darwin" else 1e6)


def _better(new: dict, old: dict) -> bool:
    """Promotion gate.

    With season CV (3+ seasons): the candidate was tuned against the
    champion's own settings on the same folds and retrained on more data, so
    it is promoted if its CV Powder Accuracy beats the raw multi-model consensus
    by 2+ points and isn't worse than the champion's settings re-scored on the
    same folds by > 0.5 point.

    Without CV: beat the raw consensus on the held-out season by 2+ points,
    and compare with the champion only on identical held-out rows."""
    if new.get("cv_powder_score") is not None:
        ref = new.get("cv_raw_consensus")
        if ref is not None and new["cv_powder_score"] < ref + 2:
            return False
        # the models' own snowfall consensus (PowderBench's baseline-consensus) is
        # the bar that counts live: don't promote a model that loses to it
        sc, on = new.get("cv_snow_consensus"), new.get("cv_powder_score_snow_rows")
        if sc is not None and on is not None and on < sc:
            return False
        # vs the champion: its own settings were re-scored on these same folds
        # and rows (an old stored CV score was measured on different data)
        champ = new.get("cv_champion_params")
        return champ is None or new["cv_powder_score"] >= champ - 0.5
    ps_n, ref = new.get("powder_score"), new.get("powder_score_raw_consensus")
    if ps_n is None or (ref is not None and ps_n < ref + 2):
        return False
    # comparable only on the very same held-out rows (same label AND size;
    # a "-tail" holdout moves as the harvest grows)
    if not old or (old.get("holdout_season"), old.get("n")) != (new.get("holdout_season"), new.get("n")):
        return True
    pin_ok = new.get("pinball", 0) <= old.get("pinball", float("inf")) * 1.02
    return ps_n >= old.get("powder_score", -1e9) - 0.5 and pin_ok


@app.command()
def watchdog(hours: float = typer.Option(4.0, help="Alert for rounds this close to cutoff")):
    """Exit 1 (and print) if an open round near its cutoff lacks our merged file."""
    from .contract import open_rounds
    from .predict import have_model
    from .submit import upstream_file

    now = datetime.now(timezone.utc)
    missing = [r for r in open_rounds(now) if r.hours_left(now) <= hours and upstream_file(r) is None
               and not (REQUIRE_BUNDLE and not have_model(r.league))]  # not competing there yet
    for r in missing:
        print(f"MISSING {r.key}: cutoff {r.cutoff:%Y-%m-%d %H:%MZ} ({r.hours_left(now):.1f} h left)")
    if missing:
        raise typer.Exit(1)
    print("all open rounds near cutoff have our file")


@app.command()
def scoreboard(out: Path = typer.Option(Path("docs/scoreboard.md"))):
    """Render our standing from PowderBench's published leaderboards."""
    from .scoreboard import render

    md = render()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text()
        a, b = "<!-- scoreboard:start -->", "<!-- scoreboard:end -->"
        if a in txt and b in txt:
            readme.write_text(txt.split(a)[0] + a + "\n" + md + "\n" + b + txt.split(b)[1])
    print(md)


if __name__ == "__main__":
    sys.exit(app())



@app.command()
def nbhd(league: str = typer.Option(...), season: int = typer.Option(0, help="0: the current season"),
         minutes: int = typer.Option(300, help="Stop starting new rounds after this long"),
         bundle: str = typer.Option("nh"),
         source: str = typer.Option("gfs", help="gfs, hrrr, gefs, wind, hsp or nbm (the module of each)")):
    """Harvest the hourly-GFS neighbourhood and timing features (nbhd.py, or hrrx.py) for
    every round of one league and season in the training table; resumes from
    the `nbhd` release and publishes back to it."""
    import time

    import pandas as pd

    from . import gefx, hrrx, hsp, nbmx, store, wndx
    from . import nbhd as nb

    nb = {"hrrr": hrrx, "gefs": gefx, "wind": wndx, "hsp": hsp, "nbm": nbmx}.get(source, nb)

    use_pb_data()
    current = season == 0  # the newest season in the table, plus the days since its last round
    if current:
        store.pull("table", store.TABLE, f"{bundle}__{league}__*.parquet")
        have = sorted(int(p.stem.rsplit("__", 1)[1]) for p in store.TABLE.glob(f"{bundle}__{league}__*.parquet"))
        if not have:
            _summary(f"nbhd {league}: no table yet")
            return
        season = have[-1]
    name = f"{bundle}__{league}__{season}.parquet"
    store.pull("table", store.TABLE, name)
    if not (store.TABLE / name).exists():
        _summary(f"nbhd {league} {season}: no table file")
        return
    rd = pd.read_parquet(store.TABLE / name, columns=["round_date"])["round_date"]
    dates = sorted(set(pd.to_datetime(rd).dt.date))
    if current:
        today = datetime.now(timezone.utc).date()
        dates = sorted({*dates, *(d.date() for d in pd.date_range(dates[-1], today))})
    dest = store.WORK / nb.TAG
    f = dest / nb.asset(league, season)
    store.pull(nb.TAG, dest, f.name)
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    seen_f = dest / f.name.replace(".parquet", ".seen.json")
    store.pull(nb.TAG, dest, seen_f.name)
    seen = set(json.loads(seen_f.read_text())) if seen_f.exists() else set()
    todo = [d for d in dates if d.isoformat() not in seen]
    t0, parts, failed = time.time(), [have] if len(have) else [], 0

    def save():
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(f, index=False, compression="zstd")
        seen_f.write_text(json.dumps(sorted(seen)))
        store.push(nb.TAG, [p for p in (f, seen_f) if p.exists()])

    last = time.time()
    for d in todo:
        if time.time() - t0 > minutes * 60:
            break
        try:
            g = nb.for_round(league, d)
        except Exception as e:  # one bad run must not lose the rest
            failed += 1
            print(f"  {d}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            if failed > 20:
                break
            continue
        if len(g) or (datetime.now(timezone.utc).date() - d).days > 3:  # an empty recent round: its run may come yet
            seen.add(d.isoformat())
        if len(g):
            parts.append(g)
        if time.time() - last > 1500:
            save()
            last = time.time()
            print(f"  saved at {d}, {len(seen)}/{len(dates)} rounds", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    save()
    rows = sum(len(p) for p in parts)
    _summary(f"{source} {league} {season}: {len(seen)}/{len(dates)} rounds done, {rows} rows, {failed} failed, "
             f"{(time.time() - t0) / 60:.0f} min")


@app.command()
def arch(league: list[str] = typer.Option(["stations", "observers", "alps", "era5"], "--league", "-l"),
         dataset: list[str] | None = typer.Option(None, "--dataset", "-d", help="Only these dynamical datasets"),
         minutes: float = typer.Option(300, help="Stop starting new units after this long"),
         part: int = typer.Option(0, help="This job's share of the missing units (0-based)"),
         parts: int = typer.Option(1, help="Jobs sharing the missing units")):
    """Harvest the every-station archive (arch.py): dynamical.org round features
    for all stations of each league, ECMWF ENS back to 2020-10. Resumable."""
    from . import arch as ar

    res = ar.run(list(league), list(dataset) if dataset else None, minutes, part=part, parts=parts)
    _summary(f"arch {','.join(league)} {','.join(dataset or ['all'])}: {res}")


@app.command()
def ukmo(hour: list[int] = typer.Option([12, 0], "--hour", help="Run hours (UTC) to harvest"),
         minutes: float = typer.Option(300, help="Stop after this long"),
         workers: int = typer.Option(32, help="Processes reading files in parallel")):
    """Harvest the Met Office global model at every station straight from the
    Met Office's open-data bucket (ukmo.py), oldest month first: the bucket
    is a 2-year rolling archive. Resumable."""
    from . import ukmo as uk

    res = uk.run(minutes, workers, tuple(hour))
    _summary(f"ukmo {','.join(map(str, hour))}Z: {res}")


@app.command()
def ecop(hour: list[int] = typer.Option([12, 0], "--hour", help="Run hours (UTC) to harvest"),
         minutes: float = typer.Option(300, help="Stop after this long"),
         workers: int = typer.Option(6, help="Threads reading forecast steps in parallel (S3 throttles above ~8)")):
    """Harvest ECMWF IFS HRES at every station straight from ECMWF's open data
    on AWS (ecop.py), back to 2023-01. Resumable."""
    from . import ecop as ec

    res = ec.run(minutes, workers, tuple(hour))
    _summary(f"ecop {','.join(map(str, hour))}Z: {res}")


@app.command("direct-windows")
def direct_windows(source: str = typer.Argument(..., help="ukmo or ecop"),
                   pull: bool = typer.Option(True, help="Fetch the harvested months from the release first")):
    """Round windows (`<var>__<model>` per station, round and horizon) from a
    direct-source harvest, saved as `<tag>__windows__12z.parquet` in its
    release, so experiments and training read one small file."""
    import importlib

    from . import store

    mod = importlib.import_module(f"dpow.{source}")
    if pull:
        mod.pull()
    w = mod.windows(12)
    p = mod.DIR / f"{mod.TAG}__windows__12z.parquet"
    w.to_parquet(p, index=False, compression="zstd")
    store.push(mod.TAG, [p])
    _summary(f"{source} windows: {len(w)} rows, {w['station_id'].nunique()} stations, "
             f"{p.stat().st_size / 1e6:.0f} MB")


@app.command()
def ana(league: str = typer.Option(...), source: str = typer.Option(..., help="hrrr or gefs"),
        minutes: int = typer.Option(300, help="Stop starting new winters after this long")):
    """Harvest daily analysis snow at every station of a league (ana.py), one
    winter at a time since the archive starts; resumes from the `nbhd` release."""
    import pandas as pd

    from . import ana as an
    from . import store
    from .contract import stations

    use_pb_data()
    if league not in an.LEAGUES[source]:
        _summary(f"ana {league} {source}: not covered")
        return
    dest = store.WORK / an.TAG
    f = dest / an.asset(league, source)
    seen_f = f.with_suffix(".done.json")
    store.pull(an.TAG, dest, f.name)
    store.pull(an.TAG, dest, seen_f.name)
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    done = set(json.loads(seen_f.read_text())) if seen_f.exists() else set()
    # the current winter is never final: harvest it again each time
    cur = str(datetime.now(timezone.utc).year - (datetime.now(timezone.utc).month < 8))
    done.discard(cur)
    if len(have):
        have = have[pd.to_datetime(have["date"]) < pd.Timestamp(f"{cur}-08-01")]
    new, done = an.harvest(league, source, stations(league), start=an.TRUTH_FROM.get(league), minutes=minutes,
                           done=done)
    out = pd.concat([have, new], ignore_index=True) if len(new) else have
    out = out.drop_duplicates(["station_id", "date"], keep="last") if len(out) else out
    dest.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f, index=False, compression="zstd")
    seen_f.write_text(json.dumps(sorted(done)))
    store.push(an.TAG, [f, seen_f])
    n_st = out["station_id"].nunique() if len(out) else 0
    _summary(f"ana {league} {source}: winters {sorted(done)}, {len(out)} station-days, {n_st} stations")
