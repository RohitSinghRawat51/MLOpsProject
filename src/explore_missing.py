from load_data import load_raw # load_data is a file 
df = load_raw()
missing =df["Global_active_power"].isna()  ##isna is a func to detect missing data

run_id = (missing != missing.shift()).cumsum() #cumsum() is for cumulative sum

gaps = missing.groupby(run_id).agg(is_missing ="first", minutes ="size")
gaps = gaps[gaps["is_missing"]]
starts = missing.index.to_series().groupby(run_id).first()
gaps["start"]= starts[gaps.index]

print("Number of gaps:", len(gaps))
print("Gaps longer than 60 minutes:", (gaps["minutes"]> 60).sum())
print()
print("Longest 5 gaps:")
print(gaps.sort_values("minutes", ascending= False).head(5)[["start","minutes"]])
print()
print("missing minutes by year:")
print(missing.groupby(df.index.year).sum())



