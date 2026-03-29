# featurestest.py
import numpy as np

def extract_features(df, idx):
    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"FALTA COLUMNA: {col}")

    features = [
        (row["close"] - prev["close"]) / prev["close"],
        row["high"] - row["low"],
        row["close"] - row["open"],
        row["high"] - max(row["open"], row["close"]),
        min(row["open"], row["close"]) - row["low"],
        row["volume"]
    ]

    return np.array(features, dtype=np.float32)
