import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ================= CONFIG =================
CSV_PATH = "dataset_tp_sl.csv"
MODEL_PATH = "model.h5"
TEST_SIZE = 0.2
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001
# ==========================================

# ---------- CARGAR DATA ----------
df = pd.read_csv(CSV_PATH)

# Columnas que usamos como features
FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return",
    "range",
    "body"
]

TARGET = "target"

# Seguridad
df = df.dropna().reset_index(drop=True)

X = df[FEATURES].values
y = df[TARGET].values

# ---------- SPLIT ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)

# ---------- ESCALADO ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ---------- MODELO ----------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------- ENTRENAMIENTO ----------
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ---------- EVALUACIÓN ----------
preds = (model.predict(X_val) > 0.5).astype(int)

print("\n===== RESULTADO REAL =====")
print("Accuracy:", np.mean(preds.flatten() == y_val))
print("Matriz de confusión:")
print(confusion_matrix(y_val, preds))
print("\nReporte:")
print(classification_report(y_val, preds))

# ---------- GUARDAR ----------
model.save(MODEL_PATH)
print(f"\nModelo guardado como {MODEL_PATH}")
