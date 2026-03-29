import numpy as np


def detect_swings(df, lookback=2):
    highs = df["high"].values
    lows = df["low"].values

    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        if highs[i] > max(highs[i-lookback:i]) and highs[i] > max(highs[i+1:i+lookback+1]):
            swing_highs.append((i, highs[i]))

        if lows[i] < min(lows[i-lookback:i]) and lows[i] < min(lows[i+1:i+lookback+1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def market_structure(swing_highs, swing_lows):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 0  # rango

    h1, h2 = swing_highs[-2][1], swing_highs[-1][1]
    l1, l2 = swing_lows[-2][1], swing_lows[-1][1]

    if h2 > h1 and l2 > l1:
        return 1    # uptrend
    if h2 < h1 and l2 < l1:
        return -1   # downtrend

    return 0


def support_resistance_zones(swing_highs, swing_lows, atr, atr_mult=0.3):
    zone = atr * atr_mult
    supports = [(l-zone, l+zone) for _, l in swing_lows[-5:]]
    resistances = [(h-zone, h+zone) for _, h in swing_highs[-5:]]
    return supports, resistances


def is_near(price, zones):
    return int(any(low <= price <= high for low, high in zones))
