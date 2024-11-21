from tensorflow import Sequential
from tensorflow import Dense, Flatten

# Example neural network for one range
def create_model():
    model = Sequential([
        Flatten(input_shape=(8, 8)),  # Flatten the 8x8 board
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(4096, activation="softmax")  # Output layer for move probabilities
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model.save("models/1500_2000_model.h5")
