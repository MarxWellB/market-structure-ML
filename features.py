import pandas as pd

def compute_features(df):
    df["return"] = df["close"].pct_change()
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["wick"] = (df["high"] - df[["open","close"]].max(axis=1)) + \
                 (df[["open","close"]].min(axis=1) - df["low"])
    df["net_move"] = df["close"].diff()
    df["momentum"] = df["net_move"].rolling(3).mean()

    return df.dropna()
