import pandas as pd
import numpy as np

# ================= CONFIG =================
CSV_PATH = "data/GBP-USD_Hour_2025-09-01_to_2025-09-30_UTC.csv"  
OUT_PATH = "dataset_state_h1.csv"

WINDOW = 60        # velas pasadas
LOOKAHEAD = 12     # velas futuras
TP_PIPS = 10
SL_PIPS = 10
PIP = 0.0001       # EURUSD
# =========================================

# ---------- CARGA ----------
df = pd.read_csv(CSV_PATH)

# Normalizar nombres de columnas
df.columns = [c.lower().strip() for c in df.columns]

# Verificación mínima
required_cols = {"open", "high", "low", "close"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Columnas faltantes. Encontradas: {df.columns}")

df = df.dropna().reset_index(drop=True)

# ---------- FEATURES ----------
def compute_features(window):
    open_ = window["open"].values
    high = window["high"].values
    low = window["low"].values
    close = window["close"].values

    returns = np.diff(close) / close[:-1]

    return {
        # volatilidad
        "ret_std": np.std(returns),
        "range_mean": np.mean(high - low),

        # estructura
        "body_mean": np.mean(np.abs(close - open_)),
        "wick_mean": np.mean((high - low) - np.abs(close - open_)),

        # direccionalidad
        "net_move": close[-1] - close[0],
        "directionality": abs(close[-1] - close[0]) / (np.sum(high - low) + 1e-9),
    }

# ---------- TARGET ----------
def compute_target(future, entry):
    tp = entry + TP_PIPS * PIP
    sl = entry - SL_PIPS * PIP

    for _, r in future.iterrows():
        if r["high"] >= tp:
            return 1
        if r["low"] <= sl:
            return 0
    return 0

# ---------- BUILD DATASET ----------
rows = []

for i in range(WINDOW, len(df) - LOOKAHEAD):
    past = df.iloc[i - WINDOW:i]
    future = df.iloc[i:i + LOOKAHEAD]

    features = compute_features(past)
    entry_price = df.iloc[i]["close"]
    target = compute_target(future, entry_price)

    features["target"] = target
    rows.append(features)

dataset = pd.DataFrame(rows)
dataset.to_csv(OUT_PATH, index=False)

print("Dataset generado:", OUT_PATH)
print("Filas:", len(dataset))
print("Winrate base:", round(dataset['target'].mean(), 4))
