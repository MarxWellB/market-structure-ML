from keras.models import load_model

model = load_model("state_model_h1.keras")
model.save_weights("state_model_h1.weights.h5")

print("Pesos exportados correctamente")
