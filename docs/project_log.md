# Project Log

## Data findings (UCI Household Electric Power Consumption)
- 2,075,259 rows, one per minute, from 2006-12-16 17:24 to 2010-11-26 21:02.
- Semicolon-separated; dates are day/month/year; missing values are written as "?".
- Missing data comes as whole rows (all 7 measurements missing together): 25,979 minutes (~1.25%).
- 71 gaps in total, only 9 longer than 60 minutes. The 5 longest (2 to 5 days each) hold most of the missing data.
- 2010 holds about two-thirds of the missing minutes, including the longest gap (7,226 min from 2010-08-17).

## Decisions
- Convert to hourly averages for forecasting.
- Fill short gaps (a few hours) by interpolation; leave long gaps empty so models are never trained or scored on invented readings.
- Meter outages are data-quality problems, not drift. Keep them out of drift tests.