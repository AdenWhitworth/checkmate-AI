#loss: 5.6417 - accuracy: 0.0451 - val_loss: 5.6630 - val_accuracy: 0.0462
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

# Function to count games by ELO range
def count_games_by_elo(pgn_file, elo_ranges):
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

# Function to calculate class weights based on ELO counts
def calculate_class_weights(elo_counts):
    total_games = sum(elo_counts.values())
    num_classes = len(elo_counts)
    return {i: total_games / (num_classes * count) for i, count in enumerate(elo_counts.values())}

# Function to extract games by ELO range
def extract_games_by_elo(pgn_file, elo_range, max_games):
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

# Build vocabulary from all games
def build_vocab(games):
    vocab = defaultdict(int)
    for game in games:
        for move in game:
            vocab[move] += 1
    move_to_id = {move: idx for idx, move in enumerate(vocab.keys())}
    id_to_move = {idx: move for move, idx in move_to_id.items()}
    return move_to_id, id_to_move

# Preprocess games for next move prediction
def preprocess_games_for_next_move(games, move_to_id):
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

# Build the transformer model
def build_transformer_model(vocab_size, max_length, embed_dim=128, num_heads=4, ff_dim=128):
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

# Training pipeline
def train_next_move_model_with_weights(pgn_file, elo_ranges, batch_size=64, epochs=10):
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
    with open("models/base_transformer_full_games_15k_games_Models/move_to_id.json", "w") as f:
        json.dump(move_to_id, f)
    with open("models/base_transformer_full_games_15k_games_Models/id_to_move.json", "w") as f:
        json.dump(id_to_move, f)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Build and compile the model
    vocab_size = len(move_to_id)
    model = build_transformer_model(vocab_size, max_length)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/base_transformer_full_games_15k_games_Models/model_checkpoint.tf",
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
    model.save("models/base_transformer_full_games_15k_games_Models/next_move_model.tf")
    return history, model, move_to_id, id_to_move


pgn_file = "../../PGN Games/partial_lichess_games_15k_filtered.pgn"
elo_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 4000)]
history, model, move_to_id, id_to_move = train_next_move_model_with_weights(pgn_file, elo_ranges)

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