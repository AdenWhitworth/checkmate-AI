#old base loss: 4.4372 - accuracy: 0.1923 - val_loss: 4.1629 - val_accuracy: 0.2326 - lr: 1.0000e-04 from output_data_v2 found in models

#Base loss: 5.5217 - accuracy: 0.0651 - val_loss: 5.2731 - val_accuracy: 0.0859 - lr: 5.0000e-05

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt

# Convert FEN to a numerical representation
def fen_to_matrix(fen):
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
                    row_array.append(piece_map.get(char, 0))  # Default to 0 for unknown pieces
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
    print(f"Preparing data from {csv_file} for ELO range {elo_range}...")
    df = pd.read_csv(csv_file)
    
    # Ensure required columns exist
    required_columns = {"FEN", "Move", "WhiteElo", "BlackElo"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file {csv_file} is missing required columns: {required_columns - set(df.columns)}")
    
    # Drop rows with NaN or invalid data
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
            avg_elo = (white_elo + black_elo) / 2  # Average ELO for the game
            feature_data.append(np.append(additional_features, [elo_range, avg_elo]))
            board_data.append(board_matrix)
            y.append(move_to_label(move))
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue  # Skip invalid rows
    
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
    board_input = Input(shape=(8, 8, 1), name="board_input")
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(board_input)
    x = MaxPooling2D((2, 2), padding="same")(x)  # Output: 4x4
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)  # Output: 2x2
    x = Conv2D(256, (2, 2), activation="relu", padding="same")(x)  # Keep spatial size at 2x2
    x = Flatten()(x)  # Flatten for fully connected layer

    features_input = Input(shape=(10,), name="features_input")  # Includes ELO range and avg ELO
    y = Dense(32, activation="relu")(features_input)

    combined = Concatenate()([x, y])
    z = Dense(256, activation="relu")(combined)
    z = Dropout(0.5)(z)
    z = Dense(4096, activation="softmax")(z)

    model = Model(inputs=[board_input, features_input], outputs=z)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Analyze label distribution
def analyze_label_distribution(y):
    label_counts = Counter(y)
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel("Labels (Moves)")
    plt.ylabel("Frequency")
    plt.title("Label Distribution")
    plt.show()

# File paths and ELO ranges
elo_ranges = {
    "Processing/15000_GAMES_FENS/base_games_fen.csv": 0.8,  # Combined range
    "Processing/5000_GAMES_FENS/greater_2000.csv": 1.0,  # High ELO
    "Processing/5000_GAMES_FENS/1500_2000.csv": 0.66,   # Mid ELO
    "Processing/5000_GAMES_FENS/1000_1500.csv": 0.33,   # Low ELO
    "Processing/5000_GAMES_FENS/less_1000.csv": 0.0     # Very Low ELO
}

# Step 1: Train base model on high ELO games
base_csv_file = "Processing/15000_GAMES_FENS/base_games_fen.csv"
print(f"Training base model on {base_csv_file}...")
x_train, x_test, y_train, y_test = prepare_data_with_elo(
    base_csv_file, elo_ranges[base_csv_file]
)

# Analyze label distribution
analyze_label_distribution(y_train)

# Separate combined data into board and feature data
board_train, feature_train = zip(*x_train)
board_test, feature_test = zip(*x_test)

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

# Save the base model
model_dir = "models/chess_CNN_unified_Models"
os.makedirs(model_dir, exist_ok=True)
base_model_file = f"{model_dir}/base_model.keras"
base_model.save(base_model_file)
print(f"Base model saved at {base_model_file}")

"""
# Fine-tune model on other ELO ranges
for csv_file, elo_range in elo_ranges.items():
    if csv_file == base_csv_file:  # Skip the base model file as it has already been trained
        continue
    
    print(f"\nFine-tuning model on {csv_file} for ELO range {elo_range}...")
    
    # Prepare data
    x_train, x_test, y_train, y_test = prepare_data_with_elo(csv_file, elo_range)

    # Separate combined data into board and feature data
    board_train, feature_train = zip(*x_train)
    board_test, feature_test = zip(*x_test)
    
    # Load the base model
    model = load_model(base_model_file)
    
    # Adjust learning rate for fine-tuning
    tf.keras.backend.set_value(model.optimizer.learning_rate, 1e-5)
    
    # Fine-tune the model
    model.fit(
        {"board_input": np.array(board_train), "features_input": np.array(feature_train)},
        y_train,
        epochs=20,  # Use fewer epochs for fine-tuning
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
    )
    
    # Evaluate and save the fine-tuned model
    test_loss, test_accuracy = model.evaluate(
        {"board_input": np.array(board_test), "features_input": np.array(feature_test)},
        y_test
    )
    print(f"Fine-tuned Test Accuracy for {csv_file}: {test_accuracy * 100:.2f}%")
    
    fine_tuned_model_file = f"{model_dir}/{csv_file.split('/')[-1].replace('.csv', '_fine_tuned.keras')}"
    model.save(fine_tuned_model_file)
    print(f"Fine-tuned model saved at {fine_tuned_model_file}")"""

