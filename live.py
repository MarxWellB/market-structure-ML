import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ================= CONFIG =================
SYMBOLS = [
    "XAUUSD",
    "XAGUSD",
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "USDCAD",
    "AUDUSD",
]

TIMEFRAME = mt5.TIMEFRAME_H1
BARS = 100

MODEL_WEIGHTS = "state_model_h1.weights.h5"

TP_PIPS = 10
SL_PIPS = 12
DEVIATION = 50
MAGIC = 999

RISK_PER_TRADE_USD = 10  # fijo para demo
COOLDOWN_BARS = 3
THRESHOLD = 0.5

PIP_VALUE = {
    "XAUUSD": 0.1,
    "XAGUSD": 0.01,
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDJPY": 0.01,
}
# =========================================


# ========== MODELO (MISMA ARQUITECTURA) ==========
def build_model():
    model = Sequential([
        Dense(64, activation="relu", input_shape=(6,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    return model


model = build_model()
model.load_weights(MODEL_WEIGHTS)
print("✅ Modelo cargado")


# ========== MT5 INIT ==========
if not mt5.initialize():
    raise RuntimeError("❌ No se pudo inicializar MT5")

account = mt5.account_info()
print(f"Cuenta: {account.login} Balance: {account.balance}")


# ========== HELPERS ==========
last_trade_bar = {s: -999 for s in SYMBOLS}


def get_features(df):
    df["return"] = df["close"].pct_change()
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["wick_mean"] = (
        (df["high"] - df[["open", "close"]].max(axis=1)) +
        (df[["open", "close"]].min(axis=1) - df["low"])
    ) / 2
    df["net_move"] = df["close"].diff()
    df["momentum"] = df["net_move"].rolling(3).mean()
    return df.dropna()


def get_lot(symbol):
    info = mt5.symbol_info(symbol)
    lot = info.volume_min
    return round(max(lot, 0.01), 2)


# ========== MAIN LOOP ==========
while True:
    for symbol in SYMBOLS:
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS)
        if rates is None or len(rates) < 50:
            continue

        df = pd.DataFrame(rates)
        df = get_features(df)

        X = df[["return", "range", "body", "wick_mean", "net_move", "momentum"]].values
        prob = model.predict(X[-1].reshape(1, -1), verbose=0)[0][0]

        bar_index = len(df)

        # ===== COOLDOWN =====
        if bar_index - last_trade_bar[symbol] < COOLDOWN_BARS:
            continue

        # ===== SEÑAL INVERTIDA =====
        if prob >= THRESHOLD:
            order_type = mt5.ORDER_TYPE_SELL
            action = "SELL"
        else:
            order_type = mt5.ORDER_TYPE_BUY
            action = "BUY"

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        volume = get_lot(symbol)

        # ===== 1️⃣ ABRIR ORDEN SIN SL/TP =====
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC,
            "comment": "AI_ENTRY",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"{symbol} ERROR OPEN: {result.retcode}")
            continue

        print(f"{symbol} TRADE OK | {action} | prob={prob:.4f}")
        last_trade_bar[symbol] = bar_index

        time.sleep(0.3)

        # ===== 2️⃣ MODIFICAR SL / TP =====
        pip = PIP_VALUE.get(symbol, 0.0001)

        if order_type == mt5.ORDER_TYPE_BUY:
            sl = result.price - SL_PIPS * pip
            tp = result.price + TP_PIPS * pip
        else:
            sl = result.price + SL_PIPS * pip
            tp = result.price - TP_PIPS * pip

        modify = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": result.order,
            "sl": sl,
            "tp": tp,
        }

        mt5.order_send(modify)

    time.sleep(30)
