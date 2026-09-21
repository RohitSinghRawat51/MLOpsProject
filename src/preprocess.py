import pandas as pd
from load_data import load_raw

MAX_GAP_HOURS = 3 #interpolating gaps up to 3; leaving longer gaps as NaN(not a Number)

def to_hourly(df):
    """Avg the minute level readings into hourly readings."""
    hourly = df.resample("h").mean()
    return hourly

def fill_short_gaps(hourly, max_gap_hours=MAX_GAP_HOURS):
    """interpolate gaps of up to max_gap_hours; leave longer gaps as NaN"""
    filled = hourly.interpolate(method="time", limit = max_gap_hours, limit_area="inside")
    return filled

if __name__ == "__main__":
    df = load_raw()
    hourly = to_hourly(df)
    filled = fill_short_gaps(hourly)

    print("Hourly shape:", filled.shape)
    print("Missing hours remaining:", filled["Global_active_power"].isna().sum())

    filled.to_csv("data/hourly_clean.csv")
    print("Saved to data/hourly_clean.csv")