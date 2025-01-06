"""
Fine-tune a Transformer model for chess move prediction across different ELO ranges.

This script:
1. Extracts games within specific ELO ranges from a PGN file.
2. Prepares datasets by tokenizing and padding move sequences.
3. Fine-tunes a pre-trained Transformer model for each ELO range.
4. Saves fine-tuned models for each range.

Results:
- 15k Games ELO 0-999: loss: 0.3692 - accuracy: 0.9224 - val_loss: 0.4100 - val_accuracy: 0.9181
- 15k Games ELO 1000-1500: loss: 0.4495 - accuracy: 0.9058 - val_loss: 0.4782 - val_accuracy: 0.9036
- 15k Games ELO 1500-2000: loss: 0.5153 - accuracy: 0.8927 - val_loss: 0.5559 - val_accuracy: 0.8888
- 15k Games ELO 2000+: loss: 0.5937 - accuracy: 0.8781 - val_loss: 0.6249 - val_accuracy: 0.8761
"""

import chess.pgn
import json
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


# Load move mappings
def load_mappings(move_to_id_path, id_to_move_path):
    """
    Load move-to-ID and ID-to-move mappings from JSON files.

    Args:
        move_to_id_path (str): Path to the JSON file mapping moves to IDs.
        id_to_move_path (str): Path to the JSON file mapping IDs to moves.

    Returns:
        tuple: A tuple containing `move_to_id` and `id_to_move` mappings.
    """
    with open(move_to_id_path, "r") as f:
        move_to_id = json.load(f)
    with open(id_to_move_path, "r") as f:
        id_to_move = json.load(f)

    id_to_move = {int(k): v for k, v in id_to_move.items()}  # Ensure keys are integers
    return move_to_id, id_to_move


# Extract games from PGN for a specific ELO range
def extract_games_by_elo_range(pgn_file, min_elo=0, max_elo=float('inf'), limit=15000):
    """
    Extract games within a specific ELO range from a PGN file.

    Args:
        pgn_file (str): Path to the PGN file.
        min_elo (int): Minimum ELO rating for filtering games.
        max_elo (int): Maximum ELO rating for filtering games.
        limit (int): Maximum number of games to extract.

    Returns:
        list: List of games, where each game is a list of SAN moves.
    """
    games = []
    with open(pgn_file, "r") as file:
        game_count = 0
        while game_count < limit:
            game = chess.pgn.read_game(file)
            if game is None:
                break

            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))

            if min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo:
                moves = [board.san(move) for move in game.mainline_moves() for board in [game.board()]]
                games.append(moves)
                game_count += 1

        print(f"Extracted {len(games)} games for ELO range {min_elo}-{max_elo}")
    return games


# Prepare tokenized and padded dataset
def prepare_data(pgn_file, min_elo, max_elo, move_to_id, max_length):
    """
    Tokenize and pad chess games for a specific ELO range.

    Args:
        pgn_file (str): Path to the PGN file.
        min_elo (int): Minimum ELO rating.
        max_elo (int): Maximum ELO rating.
        move_to_id (dict): Mapping of SAN moves to unique IDs.
        max_length (int): Maximum sequence length for padding.

    Returns:
        np.ndarray: Padded tokenized games.
    """
    games = extract_games_by_elo_range(pgn_file, min_elo, max_elo)
    tokenized_games = [[move_to_id.get(move, 0) for move in game] for game in games]
    padded_games = pad_sequences(tokenized_games, maxlen=max_length + 1, padding='post', truncating='post')
    print(f"Padded games shape: {padded_games.shape}")
    return padded_games


# Split dataset into training and validation
def prepare_train_val_data(padded_games, test_size=0.2):
    """
    Split padded games into training and validation datasets.

    Args:
        padded_games (np.ndarray): Tokenized and padded games.
        test_size (float): Proportion of the dataset to include in the validation split.

    Returns:
        tuple: Training and validation datasets (X_train, X_val, y_train, y_val).
    """
    X = padded_games[:, :-1]  # Input: all but the last token
    y = padded_games[:, 1:]   # Target: all but the first token
    return train_test_split(X, y, test_size=test_size, random_state=42)


# Fine-tune model for a specific ELO range
def fine_tune_model(base_model_path, pgn_file, min_elo, max_elo, save_path, move_to_id, max_length, batch_size=64, epochs=5):
    """
    Fine-tune a pre-trained Transformer model for a specific ELO range.

    Args:
        base_model_path (str): Path to the base model.
        pgn_file (str): Path to the PGN file.
        min_elo (int): Minimum ELO rating.
        max_elo (int): Maximum ELO rating.
        save_path (str): Path to save the fine-tuned model.
        move_to_id (dict): Mapping of SAN moves to unique IDs.
        max_length (int): Maximum sequence length for padding.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.

    Returns:
        tf.keras.callbacks.History: Training history.
    """
    model = load_model(base_model_path)
    print(f"Loaded base model with input shape: {model.input_shape}")

    # Prepare data
    padded_games = prepare_data(pgn_file, min_elo, max_elo, move_to_id, max_length)
    X_train, X_val, y_train, y_val = prepare_train_val_data(padded_games)
    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # Compile and fine-tune
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(save_path, save_best_only=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    return history


# Main process for fine-tuning
if __name__ == "__main__":
    # Paths and parameters
    pgn_file = "../../../PGN Games/pgns/partial_lichess_games_50k-450k.pgn"
    base_model_path = "../models/base_transformer_50k_games_Models/model_checkpoint.h5"
    save_model_dir = "../models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models"
    move_to_id_path = "../models/base_transformer_50k_games_Models/move_to_id.json"
    id_to_move_path = "../models/base_transformer_50k_games_Models/id_to_move.json"
    max_length = 485

    # Load mappings
    move_to_id, id_to_move = load_mappings(move_to_id_path, id_to_move_path)

    # Define ELO ranges
    elo_ranges = [
        (0, 999, f"{save_model_dir}/fine_tuned_elo_0_999.h5"),
        (1000, 1500, f"{save_model_dir}/fine_tuned_elo_1000_1500.h5"),
        (1500, 2000, f"{save_model_dir}/fine_tuned_elo_1500_2000.h5"),
        (2000, float('inf'), f"{save_model_dir}/fine_tuned_elo_2000_plus.h5")
    ]

    # Fine-tune for each ELO range
    for min_elo, max_elo, save_path in elo_ranges:
        print(f"Fine-tuning model for ELO range {min_elo}-{max_elo}")
        fine_tune_model(
            base_model_path, pgn_file, min_elo, max_elo, save_path, move_to_id, max_length
        )
        print(f"Model fine-tuned and saved to {save_path}")