# Project Log

## Data findings (UCI Household Electric Power Consumption)
- 2,075,259 rows, one per minute, from 2006-12-16 17:24 to 2010-11-26 21:02.
- Semicolon-separated; dates are day/month/year; missing values are written as "?".
- Missing data comes as whole rows (all 7 measurements missing together): 25,979 minutes (~1.25%).
- 71 gaps in total, only 9 longer than 60 minutes. The 5 longest (2 to 5 days each) hold most of the missing data.
- 2010 holds about two-thirds of the missing minutes, including the longest gap (7,226 min from 2010-08-17).

## Decisions
- Converted to hourly averages for forecasting.
- Filled short gaps (a few hours) by interpolation; leave long gaps empty so models are never trained or scored on invented readings.
- Meter outages are data-quality problems, not drift. Kept them out of drift tests.

## Resampling and gap-filling (src/preprocess.py)
- Resampled minute-level data to hourly means: 2,075,259 rows -> 34,589 hourly rows.
- Interpolated gaps up to 3 hours (time-based interpolation, inside the series only).
- Before filling: 421 missing hours. After filling short gaps: 399 missing hours remain.
- Remaining 7 gaps match the known long outages found in the minute-level analysis
  (e.g. ~116 hours in Aug 2010 corresponds to the 7,226-minute gap).
- Saved to data/hourly_clean.csv (not versioned; regenerate via src/preprocess.py).

## Feature engineering (src/features.py)
- Built from data/hourly_clean.csv (34,589 hourly rows).
- Time features: hour, day_of_week, month, is_weekend.
- Lag features: Global_active_power at 1h, 24h, 168h (1 week) ago.
- Rolling features: 24h rolling mean/std, shifted by 1h first to avoid leakage
  (a row must never see its own value in its own rolling window).
- Dropped rows without a full week of lag history: 34,589 -> 33,455 rows, 7 -> 16 columns.
- Saved to data/features.csv (not versioned; regenerate via src/features.py).


## Future extension: EIA.gov API (idea, not yet implemented)
- EIA (U.S. Energy Information Administration) offers a free API (v2, key required)
  with current hourly electricity demand data by grid/balancing authority.
- Different granularity than UCI data (grid-level demand vs single-household meter
  readings) -- cannot be concatenated with the current dataset directly.
- Possible use: a second, independent 2026 case study once the drift-detection and
  retraining pipeline works on the UCI dataset -- run the same pipeline against
  real current EIA data to test generalization.
- Not started. Revisit after Step: automated retraining + champion-challenger.

## Future extension: Time Series Foundation Models (TSFM) comparison
- 2026 research shows pre-trained, zero-shot forecasters (Chronos-2, TimesFM-3, Moirai-2)
  can forecast without training on our data, and perform comparably to trained models
  on load/energy data (arXiv:2602.10848; arXiv:2410.09487).
- Plan: after our own RF/XGBoost/LSTM models exist, add a zero-shot Chronos-2 forecast
  as an extra comparison point (no training needed, just call the pretrained model).
- Relevant to research gap: model/tool comparison, and as a check on whether a
  foundation model degrades under drift the same way our trained models do.
- Not started yet. Revisit after Step: model training (RF/XGBoost/LSTM).