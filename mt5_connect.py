import MetaTrader5 as mt5
from config import LOGIN, PASSWORD, SERVER

def connect():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        raise RuntimeError(mt5.last_error())
    return mt5
