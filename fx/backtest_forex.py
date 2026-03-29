import pandas as pd
import numpy as np
from keras.models import load_model
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atr import calculate_atr
from features.forex_features import forex_features

from atr import calculate_atr
from fx.features.features_forex import forex_features

# ================= CONFIG =================
CSV_PATH = "data/eurusd_data.csv"   # cambia por el par
MODEL_PATH = "forex_structure_model.h5"

TP_PIPS = 10
SL_PIPS = 8
PIP = 0.0001
HORIZON = 20
COOLDOWN_BARS = 5
# ========================================

# LOAD DATA
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]
if "tickvolume" in df.columns:
    df.rename(columns={"tickvolume": "volume"}, inplace=True)

# LOAD MODEL
model = load_model(MODEL_PATH)

# ATR
atr_series = calculate_atr(df)

wins = losses = no_trades = 0
profit_pips = loss_pips = 0
last_trade_i = -999

# BACKTEST LOOP
for i in range(50, len(df) - HORIZON):

    # cooldown
    if i - last_trade_i < COOLDOWN_BARS:
        continue

    atr = atr_series.iloc[i]
    if np.isnan(atr):
        continue

    feats = forex_features(df, i, atr).reshape(1, -1)
    pred = np.argmax(model.predict(feats, verbose=0)[0])

    if pred == 0:
        no_trades += 1
        continue

    entry = df.iloc[i]["close"]
    last_trade_i = i

    if pred == 1:  # BUY
        tp = entry + TP_PIPS * PIP
        sl = entry - SL_PIPS * PIP
        for j in range(i + 1, i + HORIZON):
            if df.iloc[j]["high"] >= tp:
                wins += 1
                profit_pips += TP_PIPS
                break
            if df.iloc[j]["low"] <= sl:
                losses += 1
                loss_pips += SL_PIPS
                break

    if pred == 2:  # SELL
        tp = entry - TP_PIPS * PIP
        sl = entry + SL_PIPS * PIP
        for j in range(i + 1, i + HORIZON):
            if df.iloc[j]["low"] <= tp:
                wins += 1
                profit_pips += TP_PIPS
                break
            if df.iloc[j]["high"] >= sl:
                losses += 1
                loss_pips += SL_PIPS
                break

# RESULTS
total_trades = wins + losses
print("\n===== BACKTEST FOREX =====")
print(f"Trades totales : {total_trades}")
print(f"NO TRADE       : {no_trades}")
print(f"Ganadas        : {wins}")
print(f"Perdidas       : {losses}")
print(f"Winrate        : {wins/total_trades*100:.2f}%" if total_trades else "Winrate: N/A")
print(f"Net pips       : {profit_pips - loss_pips}")
print(f"Expectancy    : {(profit_pips - loss_pips)/total_trades:.2f} pips/trade" if total_trades else "N/A")
