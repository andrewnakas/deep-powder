# Method

deep-powder is a two-level stack on the features of its sibling entries. The first part of this
page is what is new here; the rest (archives, ensemble features, reference model) is inherited
from [open-powder-ensemble](https://github.com/andrewnakas/open-powder-ensemble).

## Scoring used in development
Every model is judged on featured stations, out of fold by season (the three most recent seasons
with a real share of the rows, each held out in turn), by two rules (`src/dpow/metrics.py`):
- **board**: each round scores 100 × (1 − MAE / MAE of climatology); the figure is the mean over
  rounds. This is the leaderboard rule in force. A round with almost no snow has a tiny
  denominator, so a small forecast on a dry day costs many points.
- **pooled**: 100 × (1 − Σ|error| / Σ|climatology error|) over all rows.

## Level 1 (`src/dpow/models/learners.py`)
Each learner predicts every row of the held-out season; the predictions are published
(`oof` release) and are the stacker's training data.
- `q50`, `qlo`, `qhi`: LightGBM quantile models (10, 25, 50, 75, 90%) on all features.
- `hurdle`: LightGBM classifiers for snow above 0.05 in., at least 1 in. and at least 6 in.
- `tweedie`: LightGBM mean with a Tweedie loss. The median under-forecasts big days; the mean
  does not, and the stacker decides how much of it to use.
- `q50_ens`: the median without the Open-Meteo multi-model block, for rows where it is missing live.
- `nn`: a neural net trained on a local GPU from the published table (in development).

## Level 2 (`src/dpow/models/stacker.py`)
Per league and horizon, fitted on featured out-of-fold rows:
1. a convex blend of the level-1 point forecasts by mean absolute error, moved off the LightGBM
   median only in steps that each buy a real gain, and shrunk back toward it when round dates are few;
2. a dry rule: rows whose chance of any snow is below a threshold go to zero. The threshold is
   picked on the board rule and only accepted if the pooled score falls by no more than 0.1;
3. interval scaling: level-1 quantiles are re-centred on the blended median and their distance
   from it is scaled, per level, to the lowest pinball loss.

The reported score of the stack is nested: the stacker that predicts a season never saw it.

## No harvest
The training table is built from the siblings' public stores once a day and published
(`table` release). deep-powder makes no Open-Meteo calls.

## Inputs tried and not adopted
An input is adopted when it raises the pooled score by 0.3 or more and every held-out season
gains (`experiment.yml`).
- **NOAA National Blend of Models snowfall** (`nbmx`, 2026-11-19): 6 h ASNOW from the NBM
  archive on AWS (`noaa-nbm-grib2-pds`, from 2020-10), the 18Z D−1 run, at the station's grid
  cell and its 3 × 3 neighbourhood, per 24/48/72 h window. US stations only (76% of rows).
  +0.24 pooled with every season up (46.9 → 47.0, 43.6 → 43.8, 47.4 → 47.8); the gain is at
  24 h (36.9 → 37.7) and on 6+ in. days (MAE 4.56 → 4.47). The amounts-only variant: +0.23,
  2024 down. Harvest stays published (`nbhd` release, prefix `nbmx_`).

---

# Inherited from open-powder-ensemble

The targets, the scoring and the deterministic features are as in
[open-powder-ml](https://github.com/andrewnakas/open-powder-ml/blob/main/docs/METHOD.md). This
entry adds the following.

## Forecast archives (dynamical.org, CC BY 4.0)
- **Ensembles:** GEFS (31 members, 00Z, from 2020-10), ECMWF IFS ENS (51, from 2024-04) and
  ECMWF AIFS ENS (51, the AI ensemble, from 2025-07).
- **Single runs:** GFS (from 2021-05), ECMWF AIFS (from 2024-04) and HRRR.
- **HRRR:** the 3 km CONUS model, from 2018. It uses the 12Z D−1 runs; their 48 h reach covers
  local day D, so HRRR feeds the 24 h horizon only. It includes HRRR's own snowfall field.

Every archive is lead-correct: training rows use exactly the run a cutoff-time forecast could
have used. Deterministic runs contribute mean snow, liquid, frozen fraction and temperature
(their quantiles would say nothing).

## Ensemble features
For each round D, the forecast uses the 00Z run that a forecast made at the league's cutoff could
use:
- **Northern Hemisphere leagues:** D−1 00Z. dynamical.org publishes it well before 00Z D or
  21Z D−1.
- **Southern Hemisphere leagues** (cutoff 11Z D−1): D−2 00Z.

Training uses exactly the same rule, so there is no look-ahead.

Each member's 3-hourly output at the station's nearest grid point is turned into snow:
- **GEFS:** precipitation × the model's own percent-frozen precipitation × a snow-to-liquid ratio
  from 2 m temperature.
- **ECMWF ENS:** precipitation × a logistic rain/snow split on the **2 m wet-bulb temperature**
  (Stull 2011, from temperature and dew point) × a ratio from **850 hPa temperature**. The cloud
  layer, not the surface, sets crystal habit and density.

The 3-hour steps are summed into **station-local days** using each station's time zone (by step
midpoint), then into the cumulative windows D, D..D+1 and D..D+2.

Per ensemble and window, the features are:
- the members' mean, p10/p25/p50/p75/p90 and SD
- P(≥1″) and P(≥6″)
- the frozen fraction
- the precipitation-weighted temperature

## Model
LightGBM quantile models (p10…p90) and a P(≥6″) classifier are trained on the union of features:
- ensemble features, deterministic multi-model features where present, station metadata and
  PowderBench climatology
- a per-station bias feature, computed out-of-fold

Rows come from the ensemble archive: GEFS from 2020-10 and ECMWF ENS from 2024-04, sampled every
2nd day from November to April (June to September in the Southern Hemisphere).

So most rows carry only ensemble features. That is also the situation of non-featured network
stations live, so they stay in-distribution.

Calibration, the extrapolation guard, the promotion gate and the online bias layer are as in
open-powder-ml. The `ref_snow` reference is the median of the ensemble medians and the
deterministic consensus.

## Stacking on the public baselines
PowderBench's `baseline-*` files (NBM, HRRR, WPC percentiles, NDFD, model consensus, …) are merged
into each round hours before the cutoff, so later update passes can see them.

From our own resolved rounds, this layer picks the single baseline partner and blend weight that
cut historical MAE the most.
- It needs at least 8 rounds and at least 2% gain to switch on.
- It blends p50, and blends quantiles and P(≥6″) where the partner has them (WPC). Otherwise it
  rescales our quantiles.
- Other competitors' submissions are never used.

## Terrain
Every station gets static terrain features from the Copernicus GLO-30 DEM (30 m), computed by
`scripts/terrain.py` and stored in `data/terrain.parquet`. The method follows mountainweather/mtnwx.
- **Local terrain:** relief within 10 km, slope, aspect, and topographic position (TPI) at 500 m,
  2 km and 10 km.
- **Height above the model grid:** the station's height above the mean terrain of a ~3 km cell
  (HRRR) and of a 0.25° cell (GEFS/ECMWF/GFS). Coarse models report 2 m temperature at their
  smoothed orography, so this offset explains much of their warm, rainy bias on mountain stations.

DEM © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS
by the European Union and ESA.

## Snowpack state before the cutoff (stations league)
SNOTEL truth is the change in snow depth, so how much of a new fall shows up depends on the
pack underneath. The model gets NRCS AWDB's daily readings dated ≤ D−2 (the newest end-of-day
reading taken before a 00Z D cutoff):
- snow depth and SWE, and the pack density (SWE ÷ depth)
- new snow over the last 1 and 3 days
- the 3-day depth change
- average air temperature
- whether the ground is bare

## Late final pass
Besides the initial submission and updates on new 00Z/12Z cycles, one final pass runs 3 to
1.25 h before each cutoff. That's when the day's last usable runs have been published.
