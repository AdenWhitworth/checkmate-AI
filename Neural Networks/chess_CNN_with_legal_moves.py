#base loss: 7.9835 - accuracy: 0.0347  

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
import matplotlib.pyplot as plt

# Convert FEN to board matrix and legal moves mask
def fen_to_matrix_with_legal_moves(fen):
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,  # White pieces
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11  # Black pieces
    }
    board_matrix = np.zeros((8, 8, 12), dtype=np.float32)
    legal_moves_mask = np.zeros(4096, dtype=np.float32)  # Flattened mask for all moves

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

    # Use train_test_split to split all components together
    board_train, board_test, legal_moves_train, legal_moves_test, y_train, y_test = train_test_split(
        board_data, legal_moves_masks, y_labels, test_size=0.2, random_state=42
    )

    return board_train, legal_moves_train, y_train, board_test, legal_moves_test, y_test

# Convert UCI move to label
def move_to_label(move):
    files = "abcdefgh"
    ranks = "12345678"
    from_square = files.index(move[0]) * 8 + ranks.index(move[1])
    to_square = files.index(move[2]) * 8 + ranks.index(move[3])
    return from_square * 64 + to_square

# Create CNN model
def create_dynamic_cnn_model():
    board_input = Input(shape=(8, 8, 12), name="board_input")
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(board_input)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Flatten()(x)

    legal_moves_mask_input = Input(shape=(4096,), name="legal_moves_mask_input")

    z = Dense(128, activation="relu")(x)
    z = Dropout(0.3)(z)
    move_logits = Dense(4096, activation="linear")(z)  # Unscaled logits for all moves
    masked_output = Multiply()([move_logits, legal_moves_mask_input])
    output_probs = Softmax(name="final_output")(masked_output)

    model = Model(inputs=[board_input, legal_moves_mask_input], outputs=output_probs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Train the model
def train_model():
    csv_file = "Processing/15000_GAMES_FENS/base_games_fen.csv"

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

    # Save model
    model_dir = "models/chess_CNN_with_legal_moves"
    os.makedirs(model_dir, exist_ok=True)
    model_file = f"{model_dir}/base_model.keras"
    model.save(model_file)
    print(f"Model saved at {model_file}")

# Run the training
train_model()








