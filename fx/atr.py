import numpy as np

def calculate_atr(df, period=14):
    """
    Calcula el ATR clásico.
    Devuelve una serie del mismo tamaño que df.
    """

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr = np.maximum(
        high - low,
        np.maximum(
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        )
    )

    # ATR con min_periods para evitar NaN eternos
    atr = tr.rolling(period, min_periods=1).mean()

    return atr
