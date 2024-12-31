import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# File paths
processed_json_path = r"D:\checkmate_ai\game_phases\midgame_data.jsonl"
fens_file = "models/checkpoints2/fens.npy"
moves_file = "models/checkpoints2/moves.npy"
labels_file = "models/checkpoints2/labels.npy"
cp_evaluations_file = "models/checkpoints2/cp_evaluations.npy"
mate_evaluations_file = "models/checkpoints2/mate_evaluations.npy"
move_to_idx_file = "models/checkpoints2/move_to_idx.json"

# FEN preprocessing functions
def fen_to_tensor(fen):
    # Ensure FEN has all six fields by appending defaults for missing fields
    parts = fen.split()
    while len(parts) < 6:
        parts.append("0" if len(parts) == 4 else "1")

    board, turn, castling, en_passant, halfmove, fullmove = parts

    board_tensor = []
    for char in board:
        if char.isdigit():
            board_tensor.extend([0] * int(char))
        elif char.isalpha():
            board_tensor.append(ord(char))

    turn_tensor = [1] if turn == 'w' else [0]
    return np.array(board_tensor + turn_tensor, dtype=np.int32)

def uci_to_tensor(moves, move_map):
    return [move_map[move] for move in moves if move in move_map]

# Load and preprocess data
if all(os.path.exists(file) for file in [fens_file, moves_file, labels_file, cp_evaluations_file, mate_evaluations_file, move_to_idx_file]):
    print("Loading preprocessed data...")
    fens = np.load(fens_file)
    moves = np.load(moves_file)
    labels = np.load(labels_file)
    cp_evaluations = np.load(cp_evaluations_file)
    mate_evaluations = np.load(mate_evaluations_file)
    with open(move_to_idx_file, "r") as f:
        move_to_idx = json.load(f)
else:
    print("Preprocessing data from JSONL file...")
    fens = []
    moves = []
    labels = []
    cp_evaluations = []
    mate_evaluations = []

    # Read the JSONL file
    with open(processed_json_path, "r") as file:
        for line in tqdm(file, desc="Processing games"):
            game = json.loads(line)
            fens.append(fen_to_tensor(game["fen"]))  # FEN before the move
            moves.append(game["moves"])             # All moves leading up to the current position
            labels.append(game["next_move"])        # The next move made by the grandmaster
            cp_eval = game.get("value_cp", 0)
            cp_evaluations.append(cp_eval / 1000.0 if cp_eval is not None else 0)  # Normalize CP to [-1, 1]
            mate_eval = game.get("value_mate", 0)
            mate_evaluations.append(mate_eval if mate_eval is not None else 0)     # Keep Mate as-is

    fens = np.array(fens, dtype=np.int32)
    cp_evaluations = np.array(cp_evaluations, dtype=np.float32)
    mate_evaluations = np.array(mate_evaluations, dtype=np.float32)

    # Encode labels
    unique_moves = sorted(set(labels))
    move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
    labels = np.array([move_to_idx[label] for label in labels])

    # Encode and pad move sequences
    moves_encoded = [uci_to_tensor(seq, move_to_idx) for seq in moves]
    moves_padded = pad_sequences(moves_encoded, padding="post")
    moves = np.array(moves_padded)

    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save(fens_file, fens)
    np.save(moves_file, moves)
    np.save(labels_file, labels)
    np.save(cp_evaluations_file, cp_evaluations)
    np.save(mate_evaluations_file, mate_evaluations)
    with open(move_to_idx_file, "w") as f:
        json.dump(move_to_idx, f)

# Split data
X_fens_train, X_fens_test, X_moves_train, X_moves_test, X_cp_train, X_cp_test, X_mate_train, X_mate_test, y_train, y_test = train_test_split(
    fens, moves, cp_evaluations, mate_evaluations, labels, test_size=0.2, random_state=42
)

# Debug shapes
print("X_fens_train shape:", X_fens_train.shape)
print("X_moves_train shape:", X_moves_train.shape)
print("X_cp_train shape:", X_cp_train.shape)
print("X_mate_train shape:", X_mate_train.shape)
print("y_train shape:", y_train.shape)

def create_transformer_model(input_fen_shape, input_move_shape, num_moves):
    fen_input = Input(shape=input_fen_shape, name="fen_input")
    move_input = Input(shape=input_move_shape, name="move_input")

    # Embedding for FEN
    fen_emb = Embedding(input_dim=12 * 8 + 2, output_dim=64)(fen_input)
    fen_emb = GlobalAveragePooling1D()(fen_emb)

    # Embedding for move sequences
    move_emb = Embedding(input_dim=num_moves, output_dim=64)(move_input)
    move_emb = GlobalAveragePooling1D()(move_emb)

    # Combine embeddings
    combined = Concatenate()([fen_emb, move_emb])

    # Project to a uniform size before adding residuals
    x = Dense(128, activation="relu")(combined)

    # Transformer block
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(tf.expand_dims(x, axis=1), tf.expand_dims(x, axis=1))
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + tf.squeeze(attn_output, axis=1))

    ffn = Dense(128, activation="relu")(out1)
    ffn = Dropout(0.1)(ffn)
    ffn_output = Dense(128)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Final output for next move prediction
    x = Dense(64, activation="relu")(out2)
    x = Dropout(0.3)(x)
    next_move_output = Dense(num_moves, activation="softmax", name="next_move_output")(x)

    # Output for CP evaluation
    cp_output = Dense(32, activation="relu")(out2)
    cp_output = Dense(1, activation="linear", name="cp_output")(cp_output)

    # Output for Mate evaluation
    mate_output = Dense(32, activation="relu")(out2)
    mate_output = Dense(1, activation="linear", name="mate_output")(mate_output)

    model = Model(inputs=[fen_input, move_input], outputs=[next_move_output, cp_output, mate_output])
    return model

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    initial_lr = 1e-4
    decay_epochs = 10
    alpha = 0.1
    return initial_lr * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(np.pi * epoch / decay_epochs)))

# Initialize model
input_fen_shape = (fens.shape[1],)
input_move_shape = (moves.shape[1],)
num_moves = len(move_to_idx)

model = create_transformer_model(input_fen_shape, input_move_shape, num_moves)

model.compile(
    optimizer="adam",
    loss={
        "next_move_output": "sparse_categorical_crossentropy",
        "cp_output": "mean_squared_error",
        "mate_output": "mean_squared_error"
    },
    metrics={
        "next_move_output": "accuracy",
        "cp_output": "mae",
        "mate_output": "mae"
    }
)

# Callbacks
checkpoint = ModelCheckpoint("models/checkpoints2/model_midgame_checkpoint.h5", save_best_only=True, monitor="val_loss", mode="min")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model
print("Training model...")
history = model.fit(
    [X_fens_train, X_moves_train],
    {"next_move_output": y_train, "cp_output": X_cp_train, "mate_output": X_mate_train},
    validation_data=([X_fens_test, X_moves_test], {"next_move_output": y_test, "cp_output": X_cp_test, "mate_output": X_mate_test}),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stopping, lr_schedule]
)

# Save the final model
model.save("models/checkpoints2/model_midgame_final.h5")
print("Model training complete and saved.")
