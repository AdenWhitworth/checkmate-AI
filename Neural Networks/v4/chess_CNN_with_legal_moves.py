"""
Script to train a CNN model for chess move prediction with legal move masking.

This script:
1. Converts FEN strings into a 3D board matrix and a 1D legal moves mask.
2. Converts UCI moves into categorical labels.
3. Trains a CNN model that predicts valid chess moves by applying a mask over illegal moves.
4. Saves the trained model for further use.

Results:
- Base model:
    Loss: 7.9835
    Accuracy: 0.0347

Functions:
- fen_to_matrix_with_legal_moves: Converts FEN strings into board matrices and legal move masks.
- prepare_data_with_legal_moves: Prepares data for training, including board matrices, masks, and labels.
- move_to_label: Converts UCI moves to unique numerical labels.
- create_dynamic_cnn_model: Builds and compiles a CNN model with legal move masking.
- train_model: Trains the model and saves the trained model to disk.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Softmax, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import chess
from collections import Counter


# Convert FEN to board matrix and legal moves mask
def fen_to_matrix_with_legal_moves(fen):
    """
    Converts a FEN string into a 3D board matrix and a 1D legal moves mask.

    Args:
        fen (str): The FEN string representing the chess board state.

    Returns:
        tuple: (board_matrix, legal_moves_mask), where:
               - board_matrix is a (8, 8, 12) array encoding piece positions.
               - legal_moves_mask is a (4096,) array marking valid moves.
    """
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,  # White pieces
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11  # Black pieces
    }
    board_matrix = np.zeros((8, 8, 12), dtype=np.float32)
    legal_moves_mask = np.zeros(4096, dtype=np.float32)

    board = chess.Board(fen)
    for square, piece in board.piece_map().items():
        x, y = divmod(square, 8)
        channel = piece_map[piece.symbol()]
        board_matrix[x, y, channel] = 1.0

    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        move_index = from_square * 64 + to_square
        legal_moves_mask[move_index] = 1.0

    return board_matrix, legal_moves_mask


# Prepare the dataset
def prepare_data_with_legal_moves(csv_file):
    """
    Prepares data for training by converting FEN strings and moves into matrices, masks, and labels.

    Args:
        csv_file (str): Path to the CSV file containing FEN and move data.

    Returns:
        tuple: Training and testing splits for board matrices, legal move masks, and labels.
    """
    print(f"Preparing data from {csv_file}...")
    df = pd.read_csv(csv_file)

    board_data = []
    legal_moves_masks = []
    y_labels = []

    for index, row in df.iterrows():
        try:
            fen = row["FEN"]
            move = row["Move"]

            # Convert FEN to board matrix and legal moves mask
            board_matrix, legal_moves_mask = fen_to_matrix_with_legal_moves(fen)
            board_data.append(board_matrix)
            legal_moves_masks.append(legal_moves_mask)

            # Convert move to label
            label = move_to_label(move)
            y_labels.append(label)
        except Exception as e:
            print(f"Skipping row {index} due to error: {e}")
            continue

    if not board_data or not legal_moves_masks or not y_labels:
        raise ValueError(f"No valid data could be extracted from {csv_file}.")

    board_data = np.array(board_data)
    legal_moves_masks = np.array(legal_moves_masks)
    y_labels = np.array(y_labels)

    return train_test_split(
        board_data, legal_moves_masks, y_labels, test_size=0.2, random_state=42
    )


# Convert UCI move to label
def move_to_label(move):
    """
    Converts a UCI move string into a unique numerical label.

    Args:
        move (str): UCI move string (e.g., 'e2e4').

    Returns:
        int: A unique label representing the move.
    """
    files = "abcdefgh"
    ranks = "12345678"
    from_square = files.index(move[0]) * 8 + ranks.index(move[1])
    to_square = files.index(move[2]) * 8 + ranks.index(move[3])
    return from_square * 64 + to_square


# Create CNN model
def create_dynamic_cnn_model():
    """
    Builds and compiles a CNN model with legal move masking.

    Returns:
        Model: A compiled Keras CNN model.
    """
    board_input = Input(shape=(8, 8, 12), name="board_input")
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(board_input)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Flatten()(x)

    legal_moves_mask_input = Input(shape=(4096,), name="legal_moves_mask_input")

    z = Dense(128, activation="relu")(x)
    z = Dropout(0.3)(z)
    move_logits = Dense(4096, activation="linear")(z)
    masked_output = Multiply()([move_logits, legal_moves_mask_input])
    output_probs = Softmax(name="final_output")(masked_output)

    model = Model(inputs=[board_input, legal_moves_mask_input], outputs=output_probs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_model(csv_file, model_dir):
    """
    Trains the model on a dataset and saves it to a file.

    Args:
        csv_file (str): Path to the CSV file containing the dataset.
        model_file (str): Path to save the trained model.
    """

    # Prepare data
    board_train, legal_moves_train, y_train, board_test, legal_moves_test, y_test = prepare_data_with_legal_moves(csv_file)

    # Build model
    model = create_dynamic_cnn_model()

    # Train model
    model.fit(
        {"board_input": board_train, "legal_moves_mask_input": legal_moves_train},
        y_train,
        validation_data=(
            {"board_input": board_test, "legal_moves_mask_input": legal_moves_test},
            y_test
        ),
        epochs=30,
        batch_size=64,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ]
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(
        {"board_input": board_test, "legal_moves_mask_input": legal_moves_test},
        y_test
    )
    print(f"Fine-tuned Test Accuracy for {csv_file}: {test_accuracy * 100:.2f}%")
    print(f"Test Loss for {csv_file}: {test_loss}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_file = f"{model_dir}/base_model.keras"
    model.save(model_file)
    print(f"Model saved at {model_file}")

# Run the training process
if __name__ == "__main__":
    csv_file = "Processing/15000_GAMES_FENS/base_games_fen.csv"
    model_dir = "models"
    train_and_save_model(csv_file, model_dir)
