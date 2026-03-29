import numpy as np
from fx.features.features_forex import forex_features
from label_forex import label_forex_trade


def build_forex_dataset(df, atr_series):
    X = []
    y = []

    for i in range(50, len(df) - 20):
        atr = atr_series.iloc[i]
        if np.isnan(atr):
            continue

        features = forex_features(df, i, atr)
        label = label_forex_trade(df, i)

        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
