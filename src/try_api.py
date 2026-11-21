import pandas as pd
import requests

URL = "http://127.0.0.1:8000"
hourly = pd.read_csv("data/hourly_clean.csv", index_col=0, parse_dates=True)["Global_active_power"]

print("health  :", requests.get(f"{URL}/health").json())

t = pd.Timestamp("2010-03-10 18:00")
history = hourly[hourly.index < t].iloc[-168:]
assert not history.isna().any(), "pick a time with no meter outage in the previous week"
reply = requests.post(f"{URL}/forecast", json={"timestamp": t.isoformat(), "last_168_hours": history.tolist()}).json()
print("forecast:", reply)
print("actual  :", round(float(hourly[t]), 3))