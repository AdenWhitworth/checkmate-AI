"""
Script to train CNN models for chess move prediction based on ELO ranges.

This script:
1. Converts FEN strings to numerical matrices and additional features.
2. Converts UCI moves to categorical labels.
3. Trains a CNN model using both board state and additional features.
4. Saves trained models and evaluates their performance.

Results:
- less_1000:    Loss: 5.0660 | Accuracy: 0.1077 (finished early)
- 1000_1500:    Loss: 5.0810 | Accuracy: 0.1046 (finished early)
- 1500_2000:    Loss: 5.1264 | Accuracy: 0.1024 (finished early)
- greater_2000: Loss: 5.1905 | Accuracy: 0.0966 (finished early)

Functions:
- fen_to_matrix: Converts a FEN string into board and additional feature matrices.
- move_to_label: Converts a UCI move string (e.g., 'e2e4') to a numerical label.
- prepare_data: Prepares data for training by splitting into train and test sets.
- create_cnn_model: Creates and compiles a CNN model with dual inputs.
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def fen_to_matrix(fen):
    """
    Converts a FEN string into an 8x8 numerical board matrix and additional features.

    Args:
        fen (str): The FEN string representing the chess board.

    Returns:
        tuple: (board_matrix, additional_features), where board_matrix is an 8x8 matrix,
               and additional_features is a vector containing metadata from the FEN.
    """
    piece_map = {
        "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,  # Black pieces
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6        # White pieces
    }
    rows, turn, castling, en_passant, half_moves, full_moves = fen.split(" ")
    rows = rows.split("/")
    board_matrix = []

    # Convert board rows
    for row in rows:
        row_array = []
        for char in row:
            if char.isdigit():
                row_array.extend([0] * int(char))  # Empty squares
            else:
                row_array.append(piece_map[char])
        board_matrix.append(row_array)

    # Flatten additional features into a single vector
    additional_features = [
        1 if turn == "w" else 0,  # White's turn
        "K" in castling, "Q" in castling, "k" in castling, "q" in castling,  # Castling rights
        ord(en_passant[0]) - ord('a') if en_passant != '-' else -1,  # En passant file (-1 if none)
        int(half_moves),  # Half-move clock
        int(full_moves)   # Full-move clock
    ]

    return np.array(board_matrix, dtype=np.int8), np.array(additional_features, dtype=np.float32)

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

def prepare_data(csv_file):
    """
    Reads a CSV file and prepares data for training.

    Args:
        csv_file (str): Path to the CSV file containing FEN and move data.

    Returns:
        tuple: ([board_train, features_train], [board_test, features_test], y_train, y_test).
    """
    df = pd.read_csv(csv_file)
    board_data = []
    feature_data = []

    for fen in df["FEN"]:
        board_matrix, additional_features = fen_to_matrix(fen)
        board_data.append(board_matrix)
        feature_data.append(additional_features)
    
    # Convert to NumPy arrays
    board_data = np.expand_dims(np.array(board_data), axis=-1)  # Add channel dimension
    feature_data = np.array(feature_data)
    y = np.array([move_to_label(move) for move in df["Move"]])

    # Split each part of the input
    board_train, board_test, features_train, features_test, y_train, y_test = train_test_split(
        board_data, feature_data, y, test_size=0.2, random_state=42
    )

    return [board_train, features_train], [board_test, features_test], y_train, y_test

def create_cnn_model():
    """
    Creates and compiles a CNN model with dual inputs for board and additional features.

    Returns:
        Model: A compiled Keras CNN model.
    """
    board_input = Input(shape=(8, 8, 1), name="board_input")
    x = Conv2D(32, (3, 3), activation="relu")(board_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Flatten()(x)

    features_input = Input(shape=(8,), name="features_input")
    y = Dense(16, activation="relu")(features_input)

    combined = Concatenate()([x, y])
    z = Dense(128, activation="relu")(combined)
    z = Dropout(0.5)(z)
    z = Dense(4096, activation="softmax")(z)

    model = Model(inputs=[board_input, features_input], outputs=z)
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

    # Prepare data
    x_train, x_test, y_train, y_test = prepare_data(csv_file)

    # Define Early Stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Create and train the model
    model = create_cnn_model()
    model.fit(
        {"board_input": x_train[0], "features_input": x_train[1]},
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # Evaluate and save the model
    test_loss, test_accuracy = model.evaluate(
        {"board_input": x_test[0], "features_input": x_test[1]},
        y_test
    )
    print(f"Test Accuracy for {csv_file}: {test_accuracy * 100:.2f}%")
    print(f"Test Loss for {csv_file}: {test_loss}")

    # Ensure model directory exists
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

        
