import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# =====================
# CONFIG
# =====================
CSV_PATH = "dataset_state_h1.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 16

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(CSV_PATH)

# Target
y = df["target"].values

# Features (todo menos target)
X = df.drop(columns=["target"]).values

print(f"Muestras totales: {len(df)}")
print(f"Proporción target=1: {y.mean():.3f}")

# =====================
# TRAIN / TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    shuffle=True,
    random_state=RANDOM_STATE
)

# =====================
# SCALING
# =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================
# MODEL
# =====================
model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =====================
# TRAIN
# =====================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# =====================
# EVALUATION
# =====================
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n===== RESULTADO REAL =====")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte:")
print(classification_report(y_test, y_pred, digits=3))

# =====================
# SAVE MODEL
# =====================
model.save_weights("state_model_h1.weights.h5")

print("\nModelo guardado como state_model_h1.keras")
