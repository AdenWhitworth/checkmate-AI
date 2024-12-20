import chess.pgn
from collections import defaultdict, Counter
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, Dropout, LayerNormalization
import random

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
    with open(pgn_file, "r") as file:
        games = []
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
                    if len(moves) <= 1:  # Log games with insufficient moves
                        print(f"Short game detected: {headers} | Moves: {moves}")
                        continue
                    games.append(moves)
    return games

# Function to annotate game phases for partial games
def annotate_game_phase_for_partial(split_point):
    if split_point < 10:
        return "opening"
    elif split_point < 30:
        return "mid-game"
    else:
        return "end-game"

# Build vocabulary
def build_vocab(games):
    vocab = defaultdict(int)
    for game in games:
        for move in game:
            vocab[move] += 1
    move_to_id = {move: idx for idx, move in enumerate(vocab.keys())}
    id_to_move = {idx: move for move, idx in move_to_id.items()}
    return move_to_id, id_to_move

# Convert board state to tensor
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = (piece.piece_type - 1) + (6 if piece.color else 0)
            row, col = chess.square_rank(square), chess.square_file(square)
            tensor[row, col, layer] = 1
    return tensor.flatten()

def process_games_with_board_features(games, move_to_id):
    inputs, outputs, board_features, phases = [], [], [], []
    for game in games:
        board = chess.Board()
        tokenized_game = [move_to_id[move] for move in game]
        for i in range(1, len(tokenized_game)):
            inputs.append(tokenized_game[:i])
            outputs.append(tokenized_game[i])
            board_features.append(board_to_tensor(board))

            # Determine phase
            if i < 10:
                phases.append("opening")
            elif i < 30:
                phases.append("mid-game")
            else:
                phases.append("end-game")

            board.push_san(game[i - 1])
    return inputs, outputs, board_features, phases

# Function to dynamically generate partial games and annotate phases
def dynamic_partial_game_generator(full_games, move_to_id, max_length, batch_size):
    """
    Dynamically generates partial games with tokenized moves, board features, and phases.
    """
    while True:
        inputs, outputs, board_features, phases = [], [], [], []
        for game in full_games:
            game_length = len(game)
            if game_length <= 1:
                continue  # Skip very short games

            # Randomly select a split point for the partial game
            split_point = random.randint(1, game_length - 1)

            # Annotate the phase
            phase = annotate_game_phase_for_partial(split_point)

            # Create partial game
            partial_game = game[:split_point]

            # Use the unified `process_games_with_board_features` function
            partial_inputs, partial_outputs, partial_boards, _ = process_games_with_board_features(
                [partial_game], move_to_id
            )

            # Extend batches
            inputs.extend(partial_inputs)
            outputs.extend(partial_outputs)
            board_features.extend(partial_boards)
            phases.extend([phase] * len(partial_inputs))  # Correctly extend phases

            # Yield a batch
            if len(inputs) >= batch_size:
                # Prepare a batch
                padded_inputs = pad_sequences(inputs[:batch_size], maxlen=max_length, padding='post', dtype=np.int32)
                yield (
                    np.array(padded_inputs, dtype=np.int32),
                    np.array(board_features[:batch_size], dtype=np.float32),
                    np.array(outputs[:batch_size], dtype=np.int32),
                    np.array(phases[:batch_size], dtype=str),  # Ensure phases are included
                )
                # Remove used samples
                inputs, outputs, board_features, phases = inputs[batch_size:], outputs[batch_size:], board_features[batch_size:], phases[batch_size:]

def combined_game_generator(full_inputs, full_outputs, full_boards, full_phases, partial_gen, max_length, batch_size):
    """
    Combines preprocessed full games with dynamically generated partial games, including board features and phases,
    but targets only the next move (last time step).
    """
    while True:
        # Full game data
        full_indices = np.random.choice(len(full_inputs), size=batch_size // 2, replace=False)
        full_batch_inputs = [full_inputs[i] for i in full_indices]
        full_batch_outputs = [full_outputs[i] for i in full_indices]
        full_batch_boards = [full_boards[i] for i in full_indices]
        full_batch_phases = [full_phases[i] for i in full_indices]

        # Ensure full inputs are uniformly padded
        padded_full_inputs = pad_sequences(full_batch_inputs, maxlen=max_length, padding='post', dtype=np.int32)

        # Partial game data
        partial_batch_inputs, partial_batch_boards, partial_batch_outputs, partial_batch_phases = next(partial_gen)

        # Combine full and partial batches
        combined_inputs = np.concatenate([padded_full_inputs, partial_batch_inputs], axis=0)
        combined_boards = np.concatenate([full_batch_boards, partial_batch_boards], axis=0)
        combined_phases = np.concatenate([full_batch_phases, partial_batch_phases], axis=0)

        # For next-move prediction, targets should only include the last move
        combined_outputs = np.concatenate([full_batch_outputs, partial_batch_outputs], axis=0)

        yield (
            [np.array(combined_inputs, dtype=np.int32), np.array(combined_boards, dtype=np.float32), np.array(combined_phases, dtype=object)],
            np.array(combined_outputs, dtype=np.int32),  # Only next move as target
        )

# Transformer model with board features
def build_transformer_with_board_features_and_phases(vocab_size, max_length, embed_dim=128, num_heads=4, ff_dim=128, board_dim=768):
    # Input layers
    move_input = Input(shape=(max_length,), name="move_input")
    board_input = Input(shape=(board_dim,), name="board_input")
    phase_input = Input(shape=(1,), dtype=tf.string, name="phase_input")

    # Embedding and transformer layers for the moves
    x = Embedding(vocab_size, embed_dim, name="move_embedding")(move_input)
    x = LayerNormalization(name="move_norm")(x)

    for i in range(3):
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, name=f"multihead_attention_{i}")(x, x)
        attention_output = Dropout(0.1, name=f"attention_dropout_{i}")(attention_output)
        x = LayerNormalization(name=f"attention_norm_{i}")(x + attention_output)

        ff_output = Dense(ff_dim, activation="relu", name=f"feedforward_{i}")(x)
        ff_output = Dropout(0.1, name=f"feedforward_dropout_{i}")(ff_output)
        x = LayerNormalization(name=f"feedforward_norm_{i}")(x + ff_output)

    # Dense layers for the board input
    board_dense = Dense(ff_dim, activation="relu", name="board_dense")(board_input)

    # Phase embedding
    phase_vocab = ["opening", "mid-game", "end-game"]
    #phase_to_int = tf.keras.layers.StringLookup(vocabulary=phase_vocab, num_oov_indices=0, name="phase_lookup")
    phase_to_int = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=phase_vocab, num_oov_indices=0, name="phase_lookup")
    phase_embed = Embedding(len(phase_vocab), embed_dim, name="phase_embedding")(phase_to_int(phase_input))
    phase_embed = tf.keras.layers.Reshape((embed_dim,), name="phase_reshape")(phase_embed)

    # Align dimensions: Repeat the phase embedding and board dense vectors along the sequence dimension
    repeated_board_dense = tf.keras.layers.RepeatVector(max_length, name="repeat_board")(board_dense)
    repeated_phase_embed = tf.keras.layers.RepeatVector(max_length, name="repeat_phase")(phase_embed)

    # Concatenate along the last axis
    combined = tf.keras.layers.Concatenate(name="concat_features")([x, repeated_board_dense, repeated_phase_embed])

    # Final dense layer for prediction
    logits = Dense(vocab_size, name="output_logits")(combined)

    # Slice to focus on the last time step
    last_logits = logits[:, -1, :]  # Keep only the last time step for prediction

    # Model definition
    model = tf.keras.Model(inputs=[move_input, board_input, phase_input], outputs=last_logits, name="transformer_with_phases")
    return model


# Main processing pipeline
pgn_file = "../PGN Games/partial_lichess_games_15k_filtered.pgn"
elo_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 4000)]

# Count games for each ELO range
print("Counting games by ELO range...")
elo_counts = count_games_by_elo(pgn_file, elo_ranges)
print("ELO Counts:", elo_counts)

# Extract dataset (realistic distribution)
balanced_games = []
for elo_range in elo_ranges:
    balanced_games.extend(extract_games_by_elo(pgn_file, elo_range, max(elo_counts.values())))

# Build vocabulary
move_to_id, id_to_move = build_vocab(balanced_games)

# Save the vocabularies
with open("models/base_transformer_v2_15k_games_Models/move_to_id.json", "w") as f:
    json.dump(move_to_id, f)
with open("models/base_transformer_v2_15k_games_Models/id_to_move.json", "w") as f:
    json.dump(id_to_move, f)

# Calculate max_length based on the longest game in the dataset
max_length = max(len(game) for game in balanced_games)
print(f"Max sequence length: {max_length}")

# Training parameters
batch_size = 32
board_dim = 768
vocab_size = len(move_to_id)

# Process full games
full_inputs, full_outputs, full_boards, full_phases = process_games_with_board_features(balanced_games, move_to_id)

# Train-test split for full games
X_train_full, X_val_full, y_train_full, y_val_full, boards_train_full, boards_val_full, phases_train_full, phases_val_full = train_test_split(
    full_inputs, full_outputs, full_boards, full_phases, test_size=0.2, random_state=42
)

# Create dynamic data generators for partial games
train_partial_gen = dynamic_partial_game_generator(balanced_games, move_to_id, max_length, batch_size // 2)
val_partial_gen = dynamic_partial_game_generator(balanced_games, move_to_id, max_length, batch_size // 2)

# Combined generators (full + partial)
train_gen = combined_game_generator(
    X_train_full, y_train_full, boards_train_full, phases_train_full, train_partial_gen, max_length, batch_size
)
val_gen = combined_game_generator(
    X_val_full, y_val_full, boards_val_full, phases_val_full, val_partial_gen, max_length, batch_size
)

# Steps per epoch: Adjust to include both full and partial games
steps_per_epoch = (len(X_train_full) * 2) // batch_size  # Accounts for full + partial games
validation_steps = (len(X_val_full) * 2) // batch_size   # Same adjustment for validation

# Build and compile the model
model = build_transformer_with_board_features_and_phases(vocab_size, max_length, board_dim=board_dim)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# Callbacks: Early stopping and model checkpointing
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/base_transformer_v2_15k_games_Models/model_checkpoint",
    save_best_only=True,
    save_format="tf",
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    callbacks=[early_stopping, checkpoint_callback]
)

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

