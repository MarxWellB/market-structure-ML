import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# ================= CONFIG =================
CSV_PATH = "data/EUR-USD_Hour_2026-01-01_to_2026-01-22_UTC.csv"
MODEL_PATH = "state_model_h1.keras"

TP_PIPS = 10
SL_PIPS = 12
PIP_VALUE = 0.0001

INITIAL_BALANCE = 10000
RISK_PER_TRADE = 10
THRESHOLD = 0.5
# ==========================================


# ---------- Cargar datos ----------
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]
df = df.reset_index(drop=True)

# ---------- Features (IGUALES A TRAIN) ----------
df["return"] = df["close"].pct_change().fillna(0)
df["range"] = df["high"] - df["low"]
df["body"] = df["close"] - df["open"]
df["wick_mean"] = (
    (df["high"] - df[["open", "close"]].max(axis=1)) +
    (df[["open", "close"]].min(axis=1) - df["low"])
) / 2
df["net_move"] = df["close"] - df["close"].shift(1)
df["momentum"] = df["net_move"].rolling(3).mean()

df = df.dropna().reset_index(drop=True)

FEATURES = ["return", "range", "body", "wick_mean", "net_move", "momentum"]
X = df[FEATURES].values

# ---------- Modelo ----------
model = load_model(MODEL_PATH)
probs = model.predict(X, verbose=0).flatten()

# ---------- Backtest ----------
balance = INITIAL_BALANCE
wins = 0
losses = 0
trades = []   # 🔴 AHORA ES LISTA

for i in range(len(df) - 1):

    row = df.loc[i + 1]
    price_open = row["open"]
    high = row["high"]
    low = row["low"]
    time_entry = row.get("utc", i)

    prob = probs[i]

    # ===== SEÑAL INVERTIDA =====
    direction = -1 if prob >= THRESHOLD else 1
    side = "sell" if direction == -1 else "buy"

    if side == "buy":
        tp = price_open + TP_PIPS * PIP_VALUE
        sl = price_open - SL_PIPS * PIP_VALUE

        if high >= tp:
            pnl = RISK_PER_TRADE
            exit_price = tp
        elif low <= sl:
            pnl = -RISK_PER_TRADE
            exit_price = sl
        else:
            continue

    else:  # SELL
        tp = price_open - TP_PIPS * PIP_VALUE
        sl = price_open + SL_PIPS * PIP_VALUE

        if low <= tp:
            pnl = RISK_PER_TRADE
            exit_price = tp
        elif high >= sl:
            pnl = -RISK_PER_TRADE
            exit_price = sl
        else:
            continue

    balance += pnl
    wins += pnl > 0
    losses += pnl < 0

    pips = abs(exit_price - price_open) / PIP_VALUE

    trades.append({
        "side": side,
        "entry_time": time_entry,
        "exit_time": time_entry,
        "entry_price": price_open,
        "exit_price": exit_price,
        "pips": pips,
        "pnl": pnl
    })

# ---------- Resultados ----------
total_trades = len(trades)
winrate = wins / total_trades if total_trades > 0 else 0

print("\n===== BACKTEST REAL (SEÑAL INVERTIDA) =====")
print(f"Trades: {total_trades}")
print(f"Wins: {wins} | Losses: {losses}")
print(f"Winrate: {winrate:.2%}")
print(f"Balance inicial: ${INITIAL_BALANCE:.2f}")
print(f"Balance final: ${balance:.2f}")
print(f"Profit neto: ${balance - INITIAL_BALANCE:.2f}")

# ---------- Mejores trades ----------
print("\n===== MEJORES OPERACIONES =====")

best_buy = max((t for t in trades if t["side"] == "buy"), key=lambda x: x["pnl"], default=None)
best_sell = max((t for t in trades if t["side"] == "sell"), key=lambda x: x["pnl"], default=None)

if best_buy:
    print("\nMEJOR BUY")
    print(f"Entrada: {best_buy['entry_time']} @ {best_buy['entry_price']}")
    print(f"Salida:  {best_buy['exit_time']} @ {best_buy['exit_price']}")
    print(f"Pips:    {best_buy['pips']:.1f}")
    print(f"PnL:     ${best_buy['pnl']:.2f}")

if best_sell:
    print("\nMEJOR SELL")
    print(f"Entrada: {best_sell['entry_time']} @ {best_sell['entry_price']}")
    print(f"Salida:  {best_sell['exit_time']} @ {best_sell['exit_price']}")
    print(f"Pips:    {best_sell['pips']:.1f}")
    print(f"PnL:     ${best_sell['pnl']:.2f}")
