"""
Script to train neural networks for chess move prediction based on ELO ranges.

This script:
1. Converts FEN positions to numerical matrices for training.
2. Converts UCI moves to categorical labels.
3. Trains a neural network for each specified ELO range.
4. Saves trained models and evaluates their performance.

Results:
- less_1000:    Loss: 5.9137 | Accuracy: 0.0977
- 1000_1500:    Loss: 5.7089 | Accuracy: 0.0999
- 1500_2000:    Loss: 5.7176 | Accuracy: 0.0941
- greater_2000: Loss: 5.7024 | Accuracy: 0.0883

Functions:
- fen_to_matrix: Converts a FEN string into an 8x8 numerical matrix.
- move_to_label: Converts a UCI move string (e.g., 'e2e4') to a numerical label.
- prepare_data: Reads data from a CSV file and prepares it for training.
- create_model: Creates and compiles the neural network model.
- train_and_save_model: Trains the model for a specific dataset and saves it.
"""
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

def fen_to_matrix(fen):
    """
    Converts a FEN string into an 8x8 numerical matrix.

    Args:
        fen (str): The FEN string representing the chess board.

    Returns:
        np.ndarray: An 8x8 matrix where each piece is represented by a unique integer.
    """
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

def prepare_data(csv_file):
    """
    Reads a CSV file and prepares data for training.

    Args:
        csv_file (str): Path to the CSV file containing FEN and move data.

    Returns:
        tuple: Training and testing splits (x_train, x_test, y_train, y_test).
    """
    df = pd.read_csv(csv_file)
    x = np.array([fen_to_matrix(fen) for fen in df["FEN"]])
    y = np.array([move_to_label(move) for move in df["Move"]])
    return train_test_split(x, y, test_size=0.2, random_state=42)

def move_to_label(move):
    """
    Converts a UCI move string to a numerical label.

    Args:
        move (str): A UCI move string (e.g., 'e2e4').

    Returns:
        int: A unique label representing the move.
    """
    files = "abcdefgh"
    ranks = "12345678"
    from_square = files.index(move[0]) * 8 + ranks.index(move[1])  # 0–63
    to_square = files.index(move[2]) * 8 + ranks.index(move[3])    # 0–63
    return from_square * 64 + to_square  # Flattened index (0–4095)

def create_model():
    """
    Creates and compiles a neural network model.

    Returns:
        Sequential: A compiled Keras model.
    """
    model = Sequential([
        Flatten(input_shape=(8, 8)),  # Flatten the 8x8 board
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(4096, activation="softmax")  # Output layer for move probabilities
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_model(csv_file, model_file):
    """
    Trains the model on a dataset and saves it to a file.

    Args:
        csv_file (str): Path to the CSV file containing the dataset.
        model_file (str): Path to save the trained model.
    """
    print(f"Processing {csv_file}...")

    # Prepare the data
    x_train, x_test, y_train, y_test = prepare_data(csv_file)

    # Create the model
    model = create_model()

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy for {csv_file}: {test_accuracy * 100:.2f}%")
    print(f"Test Loss for {csv_file}: {test_loss}")

    # Save the model
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    model.save(model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    elo_ranges = {
        "Processing/5000_GAMES_FENS/less_1000.csv": "models/less_1000_model.h5",
        "Processing/5000_GAMES_FENS/1000_1500.csv": "models/1000_1500_model.h5",
        "Processing/5000_GAMES_FENS/1500_2000.csv": "models/1500_2000_model.h5",
        "Processing/5000_GAMES_FENS/greater_2000.csv": "models/greater_2000_model.h5"
    }

    for csv_file, model_file in elo_ranges.items():
        train_and_save_model(csv_file, model_file)
