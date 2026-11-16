# deep-powder

A research entry in [PowderBench](https://powderbench.com) (snowfall forecasts for mountain
stations, 24/48/72 h). It is the third entry maintained by the benchmark's author, after
[open-powder-ml](https://github.com/andrewnakas/open-powder-ml) and
[open-powder-ensemble](https://github.com/andrewnakas/open-powder-ensemble), and it is declared as
such. It uses only public data and PowderBench's public baselines, never other teams' submissions.

Status: in development, not submitting yet. No scores are claimed here until the live boards show them.

## What is different

The two earlier entries are each one LightGBM quantile model. deep-powder is a stack:

| level | what | where it trains |
|---|---|---|
| 1 | LightGBM median and quantiles on ensemble + multi-model features | GitHub Actions |
| 1 | Classifiers for "any snow", 1 in. and 6 in. | GitHub Actions |
| 1 | LightGBM mean (Tweedie loss) and an ensemble-only median | GitHub Actions |
| 1 | Neural net over per-model, per-day tokens with station embeddings | a local GPU |
| 2 | Per league and horizon: convex blend, dry-day rule, interval scaling | GitHub Actions |

Every learner is scored out of fold on the same rows by two rules: the board rule in force (mean of
per-round scores, where dry rounds weigh heavily) and the pooled rule proposed for the boards.
The dry-day rule exists because of the first one. See [docs/METHOD.md](docs/METHOD.md).

## Data

deep-powder harvests nothing itself. It reads the public feature stores of its two siblings
(release assets): open-powder-ml's `store` (Open-Meteo multi-model archive) and
open-powder-ensemble's `ens-store*` (GEFS, ECMWF ENS, AIFS ENS, GFS, AIFS, HRRR from
dynamical.org) and `snowpack` (NRCS SNOTEL). Truth is PowderBench's public truth archive.

## Running it

```bash
pip install -e '.[dev]' && scripts/checkout_powderbench.sh work/powderbench
dpow table --bundle nh              # build the training table, publish to the `table` release
dpow oof --learner q50              # one level-1 learner, out of fold -> `oof` release
dpow stack                          # fit the stacker, print the comparison table
dpow experiment --variants base     # season CV of the reference LightGBM settings
```

On GitHub: `gh workflow run pipeline.yml` runs table, all learners in parallel, then the stack.
The table needs about 12 GB of memory; do not build it on a small machine.

MIT licence.
