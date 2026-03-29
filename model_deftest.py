# model_def.py
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_model():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(6,)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy"
    )

    return model
