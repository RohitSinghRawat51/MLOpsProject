import pandas as pd

TRAIN_END = "2008-12-31 23:00:00"
VAL_END = "2009-12-31 23:00:00"
# everything after VAL_END is simulated production

def split_data(df):
    train = df.loc[:TRAIN_END]
    val = df.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=1): VAL_END]
    prod = df.loc[pd.Timestamp(VAL_END) + pd.Timedelta(hours=1):]
    return train, val, prod

if __name__ == "__main__":
    df = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)
    train, val, prod = split_data(df)

    for name, part in [("Train", train),("Validation",val),("Simulated production",prod)]:
        print(f"{name}: {part.shape[0]} rows, {part.index.min()} -> {part.index.max()}")

    train.to_csv("data/train.csv")
    val.to_csv("data/val.csv")
    prod.to_csv("data/prod.csv")
    print("\nSaved train.csv, val.csv, prod.csv to data/")
