import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam


def build_forex_model(input_dim=3, lr=0.001):
    """
    IA Forex:
    Inputs:
      [market_state, near_support, near_resistance]

    Outputs:
      0 = NO TRADE
      1 = BUY
      2 = SELL
    """

    inp = Input(shape=(input_dim,))

    x = Dense(32, activation="relu")(inp)
    x = Dropout(0.3)(x)

    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)

    out = Dense(3, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
