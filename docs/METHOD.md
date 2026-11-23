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
- **Hourly SNOTEL snowpack to the cutoff** (`hsp`, 2026-11-20): depth and SWE change from NRCS
  AWDB hourly data up to 3 h before the cutoff (the daily snowpack features stop at D−2 and miss
  overnight storms). SNOTEL stations only (34% of rows). +0.06 pooled, with 2023 and 2024 down
  (46.9 → 46.8, 43.6 → 43.3, 47.4 → 47.9); the stations league is unchanged (47.7). Harvest stays
  published (`nbhd` release, prefix `hsp_`).

## How much the Open-Meteo multi-model block is worth
- **No Open-Meteo at all** (`no_om`, 2026-11-23): every per-model column and every consensus built
  from them blanked, so what is left is dynamical.org's ensembles and GFS/HRRR/AIFS, the snowpack
  and the static columns. −8.02 pooled (45.96 → 37.94), every season down (46.9 → 35.3,
  43.6 → 37.9, 47.4 → 40.6) and every league (alps 50.7 → 40.8, observers 41.8 → 33.4,
  stations 47.7 → 40.6). The block stays; the work is replacing its models with their original
  sources (the Met Office bucket, `ukmo.py`; ECMWF open data, `ecop.py`) for every station.
- **Only the models with a free original source** (`only_direct`, 2026-11-23): ECMWF IFS, GFS,
  UKMO, NBM and HRRR kept; ICON, GEM, JMA and Météo-France blanked, the consensus rebuilt from what
  is left, and the per-day medians taken over every model blanked. −2.23 pooled (45.3, 40.6, 45.3
  by season; alps 48.3, observers 39.1, stations 46.1). ECMWF and GFS alone (`only_ecmwf_gfs`):
  −3.21. So the five direct-source models carry about three quarters of the block's 8 points.
  The other four have no free archive to train on (DWD, MSC and JMA keep a day or two; the
  Météo-France bucket is empty), so they can only be recorded from now on.
- **Coverage:** the block is on only 11–13% of stations rows and 21–26% of observers and alps rows
  in each season (the featured stations and a sample), and `ecmwf_ifs025` on none before the 2024
  season. The direct harvests cover every station on every round, so filling the block from them
  (`ecop_fill`, `direct_fill`) is the next test.

## GFS at every station (`dynom.py`, adopted 2026-11-23)
Open-Meteo's `gfs_seamless` columns are filled, where Open-Meteo gave none, with GFS read from
dynamical.org's archive (from 2021-05): the 12Z run of the day before the round at every
station's grid point, laid out and windowed exactly like the Open-Meteo models (lead i+1 for
day i). The cross-model summaries (`cons_slr`, `nmod`, …) and `ref_snow` are rebuilt on the
filled rows. GFS goes from 13% to 85% of rows.
- **`gfs_fill`:** +0.46 pooled (45.96 → 46.42), every season up (46.9 → 47.7, 43.6 → 43.9,
  47.4 → 47.7), every league (alps 50.7 → 51.2, observers 41.8 → 42.2, stations 47.7 → 48.2) and
  every horizon (24 h 36.9 → 37.5); 6+ in. MAE 4.56 → 4.49. Peak memory 10.0 GB.
- **HRRR as well** (`dyn_fill`, `ncep_hrrr_conus` from dynamical.org's HRRR archive, 7% → 31% of
  rows): +0.44, the same as GFS alone, so HRRR is harvested but not filled.
- **GFS as its own model** (`gfs_add`, `<var>__gfs_dyn` columns beside Open-Meteo's): −0.18. Filling
  the columns the model already knows is what works.
- **ECMWF HRES and the Met Office on top of GFS**, filling `ecmwf_ifs025` (from ECMWF's open data,
  `ecop.py`, 7% → 56% of rows) and `ukmo_seamless` (Met Office open data, `ukmo.py`, 11% → 34%):
  ECMWF 46.34 and the Met Office 46.31, both below GFS alone (46.42); all three 46.57
  [47.7, 44.0, 48.0], +0.15 over GFS alone with the 2023 season flat, short of the gate. Both
  archives are young (ECMWF's open data from 2023-01, the Met Office's from 2024-11), so most
  training seasons never see them. Harvested, not filled; worth another test once they cover
  more seasons.
- **The first GFS-only run read −0.41, and it was a broken test.** The direct models have no
  snowfall field, so with GFS on most rows the snowfall share among rows with a model fell below
  `gbm.snow_ready`'s 80%, and the whole snowfall family (12 features) was dropped. Experiments that
  add or fill direct models now keep the base table's snowfall decision.
- **Snowfall:** derived as precipitation × percent frozen × 0.7 cm/mm, it does not match
  Open-Meteo's GFS snowfall well enough to fill that column (140 stations, 2025-01-02..12 rounds:
  total ratio 1.39, correlation 0.69; precipitation itself 0.92 and 0.93). HRRR's follows
  Open-Meteo's (correlation 0.92) but 28% higher. Snowfall therefore stays out of the fill; it is
  kept in the units for a separate test.
- **Live:** `predict.build` reads the round's GFS run at every station and fills the same columns
  (when the league's cutoff comes before the 12Z run is out, the newest earlier 12Z run, its days
  shifted). The daily pipeline adds the newest runs to this month's and last month's units and
  rebuilds the windows file (`dpow dynom-update`), so recent training rounds are filled too.

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
