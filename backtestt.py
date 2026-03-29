# backtest.py
import pandas as pd
import numpy as np

from model_deftest import build_model
from featurestest import extract_features
from configtest import *

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("data/xagusd_data.csv")

# NORMALIZAR COLUMNAS
df.columns = [c.strip().lower() for c in df.columns]

# MT5 → Python
if "tickvolume" in df.columns:
    df.rename(columns={"tickvolume": "volume"}, inplace=True)

if "real_volume" in df.columns:
    df.rename(columns={"real_volume": "volume"}, inplace=True)

print("COLUMNAS DETECTADAS:", df.columns.tolist())

# =============================
# LOAD MODEL
# =============================
model = build_model()
model.load_weights("state_model_h1.weights.h5")

balance = INITIAL_BALANCE
equity = INITIAL_BALANCE

wins = 0
losses = 0
profit_total = 0
loss_total = 0

open_trade = None

# =============================
# BACKTEST LOOP
# =============================
for i in range(2, len(df) - 1):

    if open_trade is not None:
        # check TP / SL
        candle = df.iloc[i]

        if open_trade["type"] == "BUY":
            if candle["low"] <= open_trade["sl"]:
                loss = -SL_PIPS
                equity += loss
                loss_total += abs(loss)
                losses += 1
                open_trade = None
            elif candle["high"] >= open_trade["tp"]:
                gain = TP_PIPS
                equity += gain
                profit_total += gain
                wins += 1
                open_trade = None

        elif open_trade["type"] == "SELL":
            if candle["high"] >= open_trade["sl"]:
                loss = -SL_PIPS
                equity += loss
                loss_total += abs(loss)
                losses += 1
                open_trade = None
            elif candle["low"] <= open_trade["tp"]:
                gain = TP_PIPS
                equity += gain
                profit_total += gain
                wins += 1
                open_trade = None

        continue

    # =============================
    # DECISION
    # =============================
    features = extract_features(df, i)
    prob = float(model.predict(features.reshape(1, -1), verbose=0)[0][0])

    candle = df.iloc[i]
    price = candle["close"]

    if prob >= THRESHOLD:
        open_trade = {
            "type": "SELL",
            "entry": price,
            "tp": price - TP_PIPS * PIP_VALUE,
            "sl": price + SL_PIPS * PIP_VALUE
        }
    elif prob < (1 - THRESHOLD):
        open_trade = {
            "type": "BUY",
            "entry": price,
            "tp": price + TP_PIPS * PIP_VALUE,
            "sl": price - SL_PIPS * PIP_VALUE
        }

# =============================
# RESULTS
# =============================
total_trades = wins + losses

print("\n========== RESULTADOS ==========")
print(f"Trades totales : {total_trades}")
print(f"Ganadas        : {wins}")
print(f"Perdidas       : {losses}")
print(f"Winrate        : {wins / total_trades * 100:.2f}%")
print(f"Ganancia pips  : {profit_total}")
print(f"Pérdida pips   : {loss_total}")
print(f"Resultado neto : {profit_total - loss_total}")
