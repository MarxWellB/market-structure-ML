import pandas as pd
import numpy as np

from forex_model import build_forex_model
from dataset_forex import build_forex_dataset
from atr import calculate_atr

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("data/eurusd_data.csv")
df.columns = [c.lower() for c in df.columns]

if "tickvolume" in df.columns:
    df.rename(columns={"tickvolume": "volume"}, inplace=True)

# =====================
# FEATURES
# =====================
atr_series = calculate_atr(df)

X, y = build_forex_dataset(df, atr_series)

print("Dataset:", X.shape, y.shape)

# =====================
# MODEL
# =====================
model = build_forex_model()

model.fit(
    X, y,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    shuffle=True
)

model.save("forex_structure_model.h5")
