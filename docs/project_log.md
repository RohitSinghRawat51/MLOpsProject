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

## LSTM (src/train_lstm.py)
- Single-feature sequence model: past 24 hours of raw Global_active_power predicts
  the next hour. Windows that cross a known data gap (meter outage) are skipped.
- Scaled with MinMaxScaler fit on train only; architecture: one LSTM layer
  (32 units) + a Dense(1) output; trained 10 epochs, Adam optimizer, MSE loss.
- Result (validation): MAE 0.4106, RMSE 0.5657 -- beats the seasonal-naive baseline
  (MAE 0.5656) by ~27%, but behind Random Forest (0.3590) and XGBoost (0.3610).
- Not a fully fair comparison yet: RF/XGBoost also see hour/day/month/rolling
  features, while this LSTM only sees raw past consumption. A multivariate LSTM
  (same engineered features fed in per timestep) is a natural next improvement.

  ## Drift detection: first PSI results (src/drift.py)
- PSI vs the full 2-year training reference: Jan 2010 = 0.9677, Aug 2010 = 1.2220.
- PSI vs a same-calendar-month reference: Jan = 0.0736 (no meaningful shift), Aug = 3.8450 (higher).
- Reading: seasonality explains January but not August. Investigating per-year August values (Step 42).
- Correction: the TSFM benchmark (arXiv:2602.10848) tested Chronos-Bolt, Chronos-2, Moirai-2 and TinyTimeMixer on ERCOT hourly load; it did not test TimesFM-3.

- PSI breakdown (Step 43): in August, 3 reference buckets (each 10% of reference hours, all below 0.187 kW) are empty in Aug 2010; each costs ~0.69, together 54% of the 3.845. About 60% of reference August hours are below 0.236 kW vs under 3% in Aug 2010.
- Aug 2008 is unusually quiet (mean 0.277, median 0.208 kW vs 0.66-0.76 mean in other Augusts) and makes up half of the training-August reference. Hypothesis: the reference is atypical, not necessarily the household drifting. Testing in Step 44.
- January breakdown is healthy: every bucket contribution below 0.04, PSI 0.0736.

- Reference-year test (Step 44): Aug 2010 vs each earlier August, as PSI / KS / Wasserstein (kW). 2007: 3.550 / 0.393 / 0.333. 2008: 5.093 / 0.780 / 0.242. 2009: 0.272 / 0.230 / 0.148. Pooled 2007+2009: 1.405 / 0.199 / 0.239. Pooled all three: 2.319 / 0.395 / 0.137.
- Finding: August consumption varies strongly year to year (monthly means 0.76, 0.28, 0.66, 0.52 kW), so no single August is a fair reference. Only Aug 2009 resembles Aug 2010. The earlier hypothesis that only Aug 2008 was atypical was partly wrong: Aug 2007 also differs, in level.
- The three measures disagree on the pooled reference (lowest Wasserstein but PSI 2.3), so one textbook threshold is unreliable here. Next: measure natural year-to-year variation for each month to calibrate thresholds (Step 45).

## Natural year-to-year variation (Step 45, src/explore_variations.py)
- For each calendar month: largest PSI/KS/Wasserstein among 2007, 2008 and 2009 pairs (assumed drift-free), versus 2010 vs 2009.
- The largest PSI among the 2007-2009 pairs exceeds 0.25 in 9 of 11 months (all 11 above 0.1); biggest: Aug 5.41, Apr 2.46, Jul 1.42, Sep 1.19.
- 2010 vs 2009 PSI above 0.25 in Feb (0.394), Jul (0.322), Aug (0.272). Only Feb also exceeds the natural range on PSI and KS; no month exceeds it on Wasserstein.
- Conclusion: the earlier August alarm (PSI 1.22 / 3.85) lies within normal year-to-year variation. Feb 2010 is the only candidate for a real change.
- Design implication: use a same-calendar-month reference and thresholds calibrated from natural variation, not 0.1 / 0.25.
- Caveat: "natural max" is a max over three pairs, which overstates variation versus a single comparison.

## Does drift score predict forecast error? (Step 46, src/explore_error.py)
- Random Forest trained on 2006-2008 only, never retrained. Monthly MAE 2010 vs 2009: mean change -0.011 kW (-3%); 10 of 11 months moved by less than ~10%; Aug 2010 error dropped 31%.
- PSI-flagged months did not degrade (Feb +1%, Jul +3%, Aug -31%); the largest increases were in low-PSI months (Jun +10%, May +6%).
- Correlation of monthly drift score with error change, n=11: PSI -0.14, KS -0.42, Wasserstein -0.46 (none significant, p >= 0.16).
- Conclusion: the held-out 2010 year contains no meaningful accuracy degradation, so a PSI trigger would have retrained for no benefit. Natural data has no real drift to detect, so drift will be injected with known type, size and start (Step 47).

## Synthetic drift injector (Step 47, src/inject_drift.py)
- Natural 2010 data has no meaningful drift, so drift of known type, size and start is injected to score detectors: sudden, gradual (60-day ramp) and recurring (60-day on/off), magnitude +50% by default.
- Drift is applied to the target in hourly_clean.csv (full hourly index, gaps as NaN), then lag/rolling features are rebuilt with build_feature_table and the production slice is taken with split_data.
- Lesson: an earlier version rebuilt lags positionally on prod.csv, where outage hours are already removed; the first row after an outage got a wrong lag (1.383 vs true 1.22 in a synthetic test). Injecting before feature building avoids this.