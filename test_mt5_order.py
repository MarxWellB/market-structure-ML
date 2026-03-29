import MetaTrader5 as mt5

mt5.initialize()

symbol = "XAUUSD"
mt5.symbol_select(symbol, True)

info = mt5.symbol_info(symbol)
tick = mt5.symbol_info_tick(symbol)

print("Min lot:", info.volume_min)
print("Stop level:", info.trade_stops_level)

price = tick.ask

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": info.volume_min,   # lote mínimo REAL
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": price - info.trade_stops_level * info.point * 2,
    "tp": price + info.trade_stops_level * info.point * 2,
    "deviation": 50,
    "magic": 999,
    "comment": "TEST",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC
}

result = mt5.order_send(request)
print(result)
