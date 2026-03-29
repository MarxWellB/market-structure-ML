import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

from model_def import build_model

# ================= CONFIG =================
SYMBOLS = [
    "XAUUSD", "XAGUSD"
    
]

ALWAYS_ALLOW = {"XAUUSD", "XAGUSD"}

FOREX_SYMBOLS = {
     "USDJPY"
}

TIMEFRAME = mt5.TIMEFRAME_H1
BARS = 50

TP_PIPS = 10
SL_PIPS = 12
THRESHOLD = 30

DESIRED_LOT = 0.01
MAGIC = 999
DEVIATION = 50
COOLDOWN_MINUTES = 60
MAX_TRADES = 15

PIP_VALUE = {
    "XAUUSD": 0.1,
    "XAGUSD": 0.01,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
    "AUDJPY": 0.01,
    "USA500": 1.0,
    "NAS100": 1.0
}

last_trade_time = {}
last_candle_time = {}
# =========================================


def get_valid_lot(symbol, desired):
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    lot = max(info.volume_min, desired)
    lot = min(lot, info.volume_max)
    steps = round(lot / info.volume_step)
    return round(steps * info.volume_step, 2)


def get_min_stop(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return info.trade_stops_level * info.point


def extract_features(df):
    df["return"] = df["close"].pct_change()
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["wick"] = (
        (df["high"] - df[["open", "close"]].max(axis=1)) +
        (df[["open", "close"]].min(axis=1) - df["low"])
    ) / 2
    df["net"] = df["close"].diff()
    df["mom"] = df["net"].rolling(3).mean()

    return df.dropna().iloc[-1][
        ["return", "range", "body", "wick", "net", "mom"]
    ].values


# ========== INIT MT5 ==========
if not mt5.initialize():
    print("❌ MT5 NO INICIALIZA")
    quit()

account = mt5.account_info()
print(f"Cuenta: {account.login} | Balance: {account.balance}")

# ========== LOAD MODEL ==========
model = build_model()
model.load_weights("state_model_h1.weights.h5")
print("✅ Modelo cargado")


# ========== MAIN LOOP ==========
while True:

    positions = mt5.positions_get()
    open_symbols = {p.symbol for p in positions} if positions else set()

    if positions and len(positions) >= MAX_TRADES:
        time.sleep(300)
        continue

    for symbol in SYMBOLS:

        if symbol in open_symbols and symbol not in ALWAYS_ALLOW:
            continue

        if not mt5.symbol_select(symbol, True):
            continue

        now = datetime.now()
        if symbol in last_trade_time:
            if (now - last_trade_time[symbol]).total_seconds() < COOLDOWN_MINUTES * 60:
                continue

        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS)
        if rates is None or len(rates) < 30:
            continue

        df = pd.DataFrame(rates)
        candle_time = df.iloc[-1]["time"]

        if symbol in last_candle_time:
            if candle_time == last_candle_time[symbol]:
                continue

        features = extract_features(df)
        prob = float(model.predict(features.reshape(1, -1), verbose=0)[0][0])

        # ===== DIRECCIÓN POR MERCADO =====
        if symbol in FOREX_SYMBOLS:
            # FOREX → entrada invertida
            direction = mt5.ORDER_TYPE_BUY if prob >= THRESHOLD else mt5.ORDER_TYPE_SELL
        else:
            # METALES / ÍNDICES → normal
            direction = mt5.ORDER_TYPE_SELL if prob >= THRESHOLD else mt5.ORDER_TYPE_BUY

        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None:
            continue

        price = tick.ask if direction == mt5.ORDER_TYPE_BUY else tick.bid
        pip = PIP_VALUE.get(symbol, info.point)
        min_stop = get_min_stop(symbol)
        if min_stop is None:
            continue

        tp_dist = max(TP_PIPS * pip, min_stop)
        sl_dist = max(SL_PIPS * pip, min_stop)

        if direction == mt5.ORDER_TYPE_BUY:
            tp = price + tp_dist
            sl = price - sl_dist
        else:
            tp = price - tp_dist
            sl = price + sl_dist

        volume = get_valid_lot(symbol, DESIRED_LOT)
        if volume is None:
            continue

        for fill in (
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_RETURN
        ):
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": direction,
                "price": price,
                "deviation": DEVIATION,
                "magic": MAGIC,
                "comment": "AI_ENTRY",
                "type_filling": fill,
                "type_time": mt5.ORDER_TIME_GTC
            })
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                break

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            continue

        mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": result.order,
            "sl": sl,
            "tp": tp
        })

        last_trade_time[symbol] = now
        last_candle_time[symbol] = candle_time

        side = "BUY" if direction == mt5.ORDER_TYPE_BUY else "SELL"
        print(f"{symbol} OK | {side} | prob={prob:.3f}")

        positions = mt5.positions_get()
        open_symbols = {p.symbol for p in positions} if positions else set()
        if positions and len(positions) >= MAX_TRADES:
            break

    time.sleep(300)
