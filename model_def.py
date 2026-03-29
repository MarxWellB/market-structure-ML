from keras.models import Sequential
from keras.layers import Dense, Input

def build_model():
    model = Sequential([
        Input(shape=(6,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    return model
