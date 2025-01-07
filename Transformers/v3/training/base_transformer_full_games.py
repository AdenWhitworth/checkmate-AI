"""
Transformer-Based Chess Move Prediction

This script:
1. Filters chess games by ELO ranges from a PGN file.
2. Balances datasets using weighted sampling based on ELO ranges.
3. Builds a vocabulary of chess moves and tokenizes games.
4. Prepares training and validation datasets for a Transformer model.
5. Trains and evaluates the model for next-move prediction.
6. Saves the trained model and vocabulary mappings for future use.

Results:
- loss: 5.6417 - accuracy: 0.0451 - val_loss: 5.6630 - val_accuracy: 0.0462
"""
import chess.pgn
import json
from collections import defaultdict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def count_games_by_elo(pgn_file, elo_ranges):
    """
    Count the number of games within specified ELO ranges.

    Args:
        pgn_file (str): Path to the PGN file.
        elo_ranges (list of tuples): ELO ranges (e.g., [(0, 1000), (1000, 1500)]).

    Returns:
        dict: A dictionary mapping ELO ranges to game counts.
    """
    counts = {f"{range_[0]}-{range_[1]}": 0 for range_ in elo_ranges}
    with open(pgn_file, "r") as file:
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break
            headers = game.headers
            if "WhiteElo" in headers and "BlackElo" in headers:
                avg_elo = (int(headers["WhiteElo"]) + int(headers["BlackElo"])) // 2
                for range_ in elo_ranges:
                    if range_[0] <= avg_elo <= range_[1]:
                        counts[f"{range_[0]}-{range_[1]}"] += 1
                        break
    return counts

def calculate_class_weights(elo_counts):
    """
    Calculate class weights for balancing ELO ranges.

    Args:
        elo_counts (dict): A dictionary of ELO counts.

    Returns:
        dict: A dictionary of class weights for each ELO range.
    """
    total_games = sum(elo_counts.values())
    num_classes = len(elo_counts)
    return {i: total_games / (num_classes * count) for i, count in enumerate(elo_counts.values())}

def extract_games_by_elo(pgn_file, elo_range, max_games):
    """
    Extract games within a specific ELO range.

    Args:
        pgn_file (str): Path to the PGN file.
        elo_range (tuple): ELO range (low, high).
        max_games (int): Maximum number of games to extract.

    Returns:
        list: List of games, each as a list of SAN moves.
    """
    games = []
    with open(pgn_file, "r") as file:
        while len(games) < max_games:
            game = chess.pgn.read_game(file)
            if game is None:
                break
            headers = game.headers
            if "WhiteElo" in headers and "BlackElo" in headers:
                avg_elo = (int(headers["WhiteElo"]) + int(headers["BlackElo"])) // 2
                if elo_range[0] <= avg_elo <= elo_range[1]:
                    moves = []
                    board = game.board()
                    for move in game.mainline_moves():
                        moves.append(board.san(move))
                        board.push(move)
                    if len(moves) > 1:
                        games.append(moves)
    return games

def build_vocab(games):
    """
    Build a vocabulary mapping chess moves to IDs.

    Args:
        games (list): List of games, each as a list of SAN moves.

    Returns:
        tuple: A tuple containing `move_to_id` and `id_to_move` mappings.
    """
    vocab = defaultdict(int)
    for game in games:
        for move in game:
            vocab[move] += 1
    move_to_id = {move: idx for idx, move in enumerate(vocab.keys())}
    id_to_move = {idx: move for move, idx in move_to_id.items()}
    return move_to_id, id_to_move

def preprocess_games_for_next_move(games, move_to_id):
    """
    Prepare game data for next-move prediction.

    Args:
        games (list): List of games, each as a list of SAN moves.
        move_to_id (dict): Mapping from SAN moves to IDs.

    Returns:
        tuple: Tokenized inputs, outputs, and max sequence length.
    """
    inputs, outputs = [], []
    for game in games:
        tokenized_game = [move_to_id[move] for move in game]
        for i in range(1, len(tokenized_game)):
            inputs.append(tokenized_game[:i])
            outputs.append(tokenized_game[i])
    max_length = max(len(seq) for seq in inputs)
    inputs = pad_sequences(inputs, maxlen=max_length, padding="post")
    outputs = np.array(outputs)
    return inputs, outputs, max_length

def build_transformer_model(vocab_size, max_length, embed_dim=128, num_heads=4, ff_dim=128):
    """
    Build a transformer model for next-move prediction.

    Args:
        vocab_size (int): Size of the vocabulary.
        max_length (int): Maximum sequence length for input.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimension of feedforward layers.

    Returns:
        tf.keras.Model: Compiled transformer model.
    """
    inputs = Input(shape=(max_length,), name="move_input")
    x = Embedding(vocab_size, embed_dim, name="move_embedding")(inputs)
    x = LayerNormalization()(x)

    for _ in range(3):  # Transformer layers
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        attention_output = Dropout(0.1)(attention_output)
        x = LayerNormalization()(x + attention_output)

        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dropout(0.1)(ff_output)
        x = LayerNormalization()(x + ff_output)

    outputs = Dense(vocab_size, activation="softmax", name="move_output")(x[:, -1, :])  # Predict next move only
    model = Model(inputs, outputs, name="transformer_next_move")
    return model

def train_next_move_model_with_weights(pgn_file, elo_ranges, output_dir, batch_size=64, epochs=10):
    """
    Train a next-move prediction model using weighted ELO data.

    Args:
        pgn_file (str): Path to the PGN file.
        elo_ranges (list of tuples): ELO ranges.
        output_dir (str): Directory to save the model and vocabulary files.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.

    Returns:
        tuple: Training history, trained model, move_to_id, id_to_move.
    """
    print("Counting games by ELO range...")
    elo_counts = count_games_by_elo(pgn_file, elo_ranges)
    print("ELO Counts:", elo_counts)

    # Calculate weights for balancing ELO ranges
    class_weights = calculate_class_weights(elo_counts)
    print("Class Weights:", class_weights)

    balanced_games = []
    for idx, elo_range in enumerate(elo_ranges):
        games = extract_games_by_elo(pgn_file, elo_range, max_games=int(elo_counts[f"{elo_range[0]}-{elo_range[1]}"]))
        weighted_games = [game for game in games for _ in range(int(class_weights[idx]))]
        balanced_games.extend(weighted_games)

    move_to_id, id_to_move = build_vocab(balanced_games)
    inputs, outputs, max_length = preprocess_games_for_next_move(balanced_games, move_to_id)

    # Save vocabularies
    with open(f"{output_dir}/move_to_id.json", "w") as f:
        json.dump(move_to_id, f)
    with open(f"{output_dir}/id_to_move.json", "w") as f:
        json.dump(id_to_move, f)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Build and compile the model
    vocab_size = len(move_to_id)
    model = build_transformer_model(vocab_size, max_length)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{output_dir}/model_checkpoint.tf",
        save_best_only=True,
        save_format="tf",
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    # Save the model
    model.save(f"{output_dir}/next_move_model.tf")
    return history, model, move_to_id, id_to_move

def plot_training_history(history):
    """
    Plot training and validation loss and accuracy from the model training history.

    Args:
        history (tf.keras.callbacks.History): Training history object returned by the model's `fit` method.
    """
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Define file paths and ELO ranges
    pgn_file = "../../../PGN Games/pgns/partial_lichess_games_15k_filtered.pgn"
    elo_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 4000)]
    output_dir = "../models/base_transformer_full_games_15k_games_Models"

    # Train the next-move prediction model
    history, model, move_to_id, id_to_move = train_next_move_model_with_weights(pgn_file, elo_ranges, output_dir)

    # Plot training results
    plot_training_history(history)
