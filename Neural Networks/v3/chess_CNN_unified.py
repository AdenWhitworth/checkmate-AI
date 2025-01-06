"""
Script to train a dynamic CNN model for chess move prediction with ELO integration.

This script:
1. Converts FEN strings into board matrices and additional features.
2. Converts UCI moves into categorical labels.
3. Trains a dynamic CNN model using high ELO games as a base dataset.
4. Implements callbacks like EarlyStopping and ReduceLROnPlateau for efficient training.
5. Saves the trained base model for further use.

Results:
- Base model:
    Loss: 5.5217
    Accuracy: 0.0651
    Validation Loss: 5.2731
    Validation Accuracy: 0.0859
    Learning Rate: 5.0000e-05

Functions:
- fen_to_matrix: Converts a FEN string into numerical board and feature matrices.
- move_to_label: Converts a UCI move (e.g., 'e2e4') into a unique numerical label.
- prepare_data_with_elo: Prepares data for training, including ELO integration.
- create_dynamic_cnn_model: Creates and compiles a dynamic CNN model with ELO features.
- analyze_label_distribution: Visualizes label distribution for debugging and analysis.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt


# Convert FEN to numerical representation
def fen_to_matrix(fen):
    """
    Converts a FEN string into an 8x8 board matrix and additional features.

    Args:
        fen (str): FEN string representing the chess board and game state.

    Returns:
        tuple: (board_matrix, additional_features), where `board_matrix` is an 8x8 array
               and `additional_features` is a vector with metadata from the FEN string.
    """
    try:
        piece_map = {
            "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
            "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6
        }
        rows, turn, castling, en_passant, half_moves, full_moves = fen.split(" ")
        rows = rows.split("/")
        board_matrix = []

        for row in rows:
            row_array = []
            for char in row:
                if char.isdigit():
                    row_array.extend([0] * int(char))
                else:
                    row_array.append(piece_map.get(char, 0))
            board_matrix.append(row_array)

        additional_features = [
            1 if turn == "w" else 0,
            "K" in castling, "Q" in castling, "k" in castling, "q" in castling,
            ord(en_passant[0]) - ord('a') if en_passant != '-' else -1,
            int(half_moves),
            int(full_moves)
        ]

        return np.array(board_matrix, dtype=np.int8), np.array(additional_features, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error processing FEN string {fen}: {e}")


# Convert UCI move to label
def move_to_label(move):
    """
    Converts a UCI move string into a unique numerical label.

    Args:
        move (str): UCI move string (e.g., 'e2e4').

    Returns:
        int: A unique label representing the move.
    """
    try:
        files = "abcdefgh"
        ranks = "12345678"
        from_square = files.index(move[0]) * 8 + ranks.index(move[1])
        to_square = files.index(move[2]) * 8 + ranks.index(move[3])
        return from_square * 64 + to_square
    except Exception as e:
        raise ValueError(f"Error processing move {move}: {e}")


# Prepare the dataset
def prepare_data_with_elo(csv_file, elo_range):
    """
    Prepares data for training, including board matrices, features, and labels.

    Args:
        csv_file (str): Path to the input CSV file.
        elo_range (float): Normalized ELO range to include as a feature.

    Returns:
        tuple: Training and testing splits for board matrices, features, and labels.
    """
    print(f"Preparing data from {csv_file} for ELO range {elo_range}...")
    df = pd.read_csv(csv_file)

    required_columns = {"FEN", "Move", "WhiteElo", "BlackElo"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file {csv_file} is missing required columns: {required_columns - set(df.columns)}")

    df.dropna(subset=["FEN", "Move", "WhiteElo", "BlackElo"], inplace=True)

    board_data = []
    feature_data = []
    y = []

    for _, row in df.iterrows():
        try:
            fen = row["FEN"]
            move = row["Move"]
            white_elo = int(row["WhiteElo"])
            black_elo = int(row["BlackElo"])

            board_matrix, additional_features = fen_to_matrix(fen)
            avg_elo = (white_elo + black_elo) / 2
            feature_data.append(np.append(additional_features, [elo_range, avg_elo]))
            board_data.append(board_matrix)
            y.append(move_to_label(move))
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue

    if not board_data or not feature_data or not y:
        raise ValueError(f"No valid data could be extracted from {csv_file}.")

    board_data = np.expand_dims(np.array(board_data), axis=-1)
    feature_data = np.array(feature_data)
    y = np.array(y)

    return train_test_split(
        list(zip(board_data, feature_data)),  # Combine board and feature data
        y,
        test_size=0.2,
        random_state=42
    )


# Create CNN model
def create_dynamic_cnn_model():
    """
    Creates and compiles a dynamic CNN model with dual inputs for board state and features.

    Returns:
        Model: Compiled Keras CNN model.
    """
    board_input = Input(shape=(8, 8, 1), name="board_input")
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(board_input)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Flatten()(x)

    features_input = Input(shape=(10,), name="features_input")
    y = Dense(32, activation="relu")(features_input)

    combined = Concatenate()([x, y])
    z = Dense(256, activation="relu")(combined)
    z = Dropout(0.5)(z)
    z = Dense(4096, activation="softmax")(z)

    model = Model(inputs=[board_input, features_input], outputs=z)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# Analyze label distribution
def analyze_label_distribution(y):
    """
    Visualizes the distribution of labels in the dataset.

    Args:
        y (list): List of labels (moves).
    """
    label_counts = Counter(y)
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel("Labels (Moves)")
    plt.ylabel("Frequency")
    plt.title("Label Distribution")
    plt.show()

def train_and_save_model(csv_file, model_dir):
    """
    Trains the model on a dataset and saves it to a file.

    Args:
        csv_file (str): Path to the CSV file containing the dataset.
        model_file (str): Path to save the trained model.
    """
    # Prepare data
    print(f"Training base model on {csv_file}...")
    elo_range = 0.8  # Base ELO range
    x_train, x_test, y_train, y_test = prepare_data_with_elo(csv_file, elo_range)

    # Analyze label distribution
    analyze_label_distribution(y_train)

    # Extract board and feature data
    board_train, feature_train = zip(*x_train)
    board_test, feature_test = zip(*x_test)

    # Create and train the model
    base_model = create_dynamic_cnn_model()
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    base_model.fit(
        {"board_input": np.array(board_train), "features_input": np.array(feature_train)},
        y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler]
    )

     # Evaluate and save the fine-tuned model
    test_loss, test_accuracy = base_model.evaluate(
        {"board_input": np.array(board_test), "features_input": np.array(feature_test)},
        y_test
    )
    print(f"Fine-tuned Test Accuracy for {csv_file}: {test_accuracy * 100:.2f}%")
    print(f"Test Loss for {csv_file}: {test_loss}")

    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    model_file = f"{model_dir}/base_model.keras"
    base_model.save(model_file)
    print(f"Base model saved at {model_file}")

# Main script
if __name__ == "__main__":
    # File paths and ELO ranges
    base_csv_file = "Processing/15000_GAMES_FENS/base_games_fen.csv"
    output_dir = "models"

    train_and_save_model(base_csv_file, output_dir)