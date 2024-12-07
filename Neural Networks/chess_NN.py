#less_1000 loss: 5.9137 - accuracy: 0.0977
#1000_1500 loss: 5.7089 - accuracy: 0.0999
#1500_2000 loss: 5.7176 - accuracy: 0.0941
#greater_2000: loss: 5.7024 - accuracy: 0.0883

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

# Convert FEN to a numerical representation
def fen_to_matrix(fen):
    piece_map = {
        "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,  # Black pieces
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6        # White pieces
    }
    rows = fen.split(" ")[0].split("/")
    board_matrix = []

    for row in rows:
        row_array = []
        for char in row:
            if char.isdigit():
                row_array.extend([0] * int(char))  # Empty squares
            else:
                row_array.append(piece_map[char])
        board_matrix.append(row_array)

    return np.array(board_matrix, dtype=np.int8)

# Prepare the dataset
def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    x = np.array([fen_to_matrix(fen) for fen in df["FEN"]])
    y = np.array([move_to_label(move) for move in df["Move"]])  # Implement move_to_label
    return train_test_split(x, y, test_size=0.2, random_state=42)

# Convert UCI move to label (e.g., 'e2e4' to index)
def move_to_label(move):
    files = "abcdefgh"
    ranks = "12345678"
    from_square = files.index(move[0]) * 8 + ranks.index(move[1])  # 0–63 for 'e2'
    to_square = files.index(move[2]) * 8 + ranks.index(move[3])    # 0–63 for 'e4'
    return from_square * 64 + to_square  # Flattened index (0–4095)

# Create the model
def create_model():
    model = Sequential([
        Flatten(input_shape=(8, 8)),  # Flatten the 8x8 board
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(4096, activation="softmax")  # Output layer for move probabilities
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Train and evaluate for each ELO range
elo_ranges = {
    "Processing/5000_GAMES_FENS/less_1000.csv": "models/chess_NN_Models/less_1000_model.h5",
    "Processing/5000_GAMES_FENS/1000_1500.csv": "models/chess_NN_Models/1000_1500_model.h5",
    "Processing/5000_GAMES_FENS/1500_2000.csv": "models/chess_NN_Models/1500_2000_model.h5",
    "Processing/5000_GAMES_FENS/greater_2000.csv": "models/chess_NN_Models/greater_2000_model.h5"
}

for csv_file, model_file in elo_ranges.items():
    print(f"Processing {csv_file}...")
    
    # Prepare data
    x_train, x_test, y_train, y_test = prepare_data(csv_file)
    
    # Create and train the model
    model = create_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate and save the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy for {csv_file}: {test_accuracy * 100:.2f}%")
    
    model.save(model_file)
    print(f"Model saved to {model_file}")
