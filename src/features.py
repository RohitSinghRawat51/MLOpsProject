import pandas as pd

TARGET = "Global_active_power"

def add_time_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek # 0 = Monday 
    df["month"]=df.index.month
    df["is_weekend"] = (df["day_of_week"] >=5).astype(int)
    return df

def add_lag_features(df, target= TARGET, lags =(1, 24, 168)):
    df= df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}h"] = df[target].shift(lag)
    return df

def add_rolling_features(df, target = TARGET, window=24):
    df = df.copy()
    #shift(1) first so today's row never sees today's own value
    df[f"{target}_roll_mean_{window}h"] = df[target].shift(1).rolling(window).mean()
    df[f"{target}_roll_std_{window}h"] = df[target].shift(1).rolling(window).std()
    return df

def build_feature_table(df):
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna() # first ~168 rows won't have a full week of lag history
    return df

if __name__ == "__main__":
    hourly = pd.read_csv("data/hourly_clean.csv",index_col=0, parse_dates = True)
    features = build_feature_table(hourly)

    print("Shape before feature engineering:", hourly.shape)
    print("Shape after feature engineering:",features.shape)
    print()
    print("Columns:",list(features.columns))
    print()
    print(features.head())

    features.to_csv("data/features.csv")
    print("\nSaved to data/features.csv")
