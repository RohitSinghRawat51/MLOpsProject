import pandas as pd
Raw_Path = "data/household_power_consumption.txt"

def load_raw(path = Raw_Path):

    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)

    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format = "%d/%m/%Y %H:%M:%S"
        )

    df = df.drop(columns=["Date","Time"])
    df = df.set_index("timestamp").sort_index()
    return df

if __name__ == "__main__":
    df=load_raw()
    print("Shape:", df.shape)
    print("First Timestamp:", df.index.min())
    print("Last Timestamp:", df.index.max())
    print()
    print(df.dtypes)
    print()
    print("Missing values per column:")
    print(df.isna().sum())
    print()
    print(df.describe().T)
