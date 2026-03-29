import MetaTrader5 as mt5
import math

def normalize_lot(lot, step):
    return math.floor(lot / step) * step

import math

def get_valid_lot(symbol, desired_lot):
    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    min_lot = info.volume_min
    max_lot = info.volume_max
    step = info.volume_step

    # Ajustar al rango permitido
    lot = max(min_lot, min(desired_lot, max_lot))

    # Ajustar al step correcto
    steps = round(lot / step)
    lot = steps * step

    # Redondeo final para evitar decimales raros
    lot = round(lot, int(abs(math.log10(step))))

    return lot