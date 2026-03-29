import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# ================= CONFIG =================
CSV_PATH = "data/USA500.IDX-USD_Hour_2025-10-01_to_2025-10-31_UTC.csv"
MODEL_PATH = "state_model_h1.keras"

TP_PIPS = 10
SL_PIPS = 12
PIP_VALUE = 0.0001

INITIAL_BALANCE = 500
RISK_PER_TRADE = 30
THRESHOLD = 0.5

COOLDOWN_BARS = 3  # SOLO PARA OBSERVAR SOBRE-EJECUCIÓN
# ==========================================


# ---------- Cargar datos ----------
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]
df = df.reset_index(drop=True)

# ---------- Features ----------
df["return"] = df["close"].pct_change().fillna(0)
df["range"] = df["high"] - df["low"]
df["body"] = df["close"] - df["open"]
df["wick_mean"] = ((df["high"] - df[["open","close"]].max(axis=1)) +
                    (df[["open","close"]].min(axis=1) - df["low"])) / 2
df["net_move"] = df["close"] - df["close"].shift(1)
df["momentum"] = df["net_move"].rolling(3).mean()

df = df.dropna().reset_index(drop=True)

FEATURES = ["return", "range", "body", "wick_mean", "net_move", "momentum"]
X = df[FEATURES].values

# ---------- Modelo ----------
model = load_model(MODEL_PATH)
probs = model.predict(X, verbose=0).flatten()

# ---------- Estado ----------
balance = INITIAL_BALANCE
position = None
last_trade_index = -999

trade_log = []

wins = 0
losses = 0


# ---------- Simulación secuencial ----------
for i in range(len(df) - 1):

    time = df.loc[i + 1, "utc"] if "utc" in df.columns else i
    open_price = df.loc[i + 1, "open"]
    high = df.loc[i + 1, "high"]
    low = df.loc[i + 1, "low"]

    prob = probs[i]

    # Señal INVERTIDA (como ya validaste)
    side = "SELL" if prob >= THRESHOLD else "BUY"

    # -------- BLOQUEOS --------
    if position is not None:
        trade_log.append({
            "time": time,
            "action": side,
            "status": "BLOCKED",
            "reason": "POSITION_OPEN",
            "prob": prob
        })
        continue

    if i - last_trade_index < COOLDOWN_BARS:
        trade_log.append({
            "time": time,
            "action": side,
            "status": "BLOCKED",
            "reason": "COOLDOWN",
            "prob": prob
        })
        continue

    # -------- EJECUCIÓN --------
    position = side
    entry_price = open_price
    entry_index = i
    last_trade_index = i

    if side == "BUY":
        tp = entry_price + TP_PIPS * PIP_VALUE
        sl = entry_price - SL_PIPS * PIP_VALUE
        if high >= tp:
            result = "WIN"
            pnl = RISK_PER_TRADE
        elif low <= sl:
            result = "LOSS"
            pnl = -RISK_PER_TRADE
        else:
            position = None
            continue

    else:
        tp = entry_price - TP_PIPS * PIP_VALUE
        sl = entry_price + SL_PIPS * PIP_VALUE
        if low <= tp:
            result = "WIN"
            pnl = RISK_PER_TRADE
        elif high >= sl:
            result = "LOSS"
            pnl = -RISK_PER_TRADE
        else:
            position = None
            continue

    balance += pnl
    wins += result == "WIN"
    losses += result == "LOSS"

    trade_log.append({
        "time": time,
        "action": side,
        "status": result,
        "entry": entry_price,
        "tp": tp,
        "sl": sl,
        "prob": prob,
        "balance": balance
    })

    position = None


# ---------- RESULTADOS ----------
log_df = pd.DataFrame(trade_log)

print("\n===== SIMULACIÓN REALISTA =====")
print(f"Balance inicial: ${INITIAL_BALANCE}")
print(f"Balance final:   ${balance}")
print(f"Ganancia neta:   ${balance - INITIAL_BALANCE}")
print(f"Wins: {wins} | Losses: {losses}")

print("\n===== RESUMEN DE EJECUCIÓN =====")
print(log_df["status"].value_counts())

print("\n===== BLOQUEOS =====")
print(log_df[log_df["status"] == "BLOCKED"]["reason"].value_counts())

# Guardar log completo
log_df.to_csv("trade_audit_log.csv", index=False)
print("\nArchivo generado: trade_audit_log.csv")
