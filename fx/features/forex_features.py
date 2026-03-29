import numpy as np
from features.structure_features import (
    detect_swings,
    market_structure,
    support_resistance_zones,
    is_near
)


def forex_features(df, idx, atr):
    context = df.iloc[:idx+1]

    swing_highs, swing_lows = detect_swings(context)
    structure = market_structure(swing_highs, swing_lows)

    supports, resistances = support_resistance_zones(
        swing_highs, swing_lows, atr
    )

    price = df.iloc[idx]["close"]

    near_support = is_near(price, supports)
    near_resistance = is_near(price, resistances)

    return np.array([
        structure,        # -1, 0, 1
        near_support,     # 0 / 1
        near_resistance   # 0 / 1
    ], dtype=np.float32)
