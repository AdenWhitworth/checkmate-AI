#loss: 1.8428 - accuracy: 0.4359 - val_loss: 1.7486 - val_accuracy: 0.4484
#loss: 1.7627 - accuracy: 0.4420 - val_loss: 1.7210 - val_accuracy: 0.4530
#loss: 2.5458 - next_move_output_loss: 1.6852 - outcome_output_loss: 0.8606 - next_move_output_accuracy: 0.4622 - outcome_output_accuracy: 0.5014 - val_loss: 2.5144 - val_next_move_output_loss: 1.6496 - val_outcome_output_loss: 0.8648 - val_next_move_output_accuracy: 0.4698 - val_outcome_output_accuracy: 0.4936
import pandas as pd
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
processed_json_path = r"D:\checkmate_ai\game_phases\open_data.json"
fens_file = "models/checkpoints3/fens.npy"
moves_file = "models/checkpoints3/moves.npy"
labels_file = "models/checkpoints3/labels.npy"
game_outcomes_file = "models/checkpoints3/game_outcomes.npy"
move_to_idx_file = "models/checkpoints3/move_to_idx.json"

# FEN preprocessing functions
def fen_to_tensor(fen):
    board, turn, _, _, _, _ = fen.split()
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

# Check if preprocessed files exist
if all(os.path.exists(file) for file in [fens_file, moves_file, labels_file, game_outcomes_file, move_to_idx_file]):
    print("Loading preprocessed data...")
    fens = np.load(fens_file)
    moves = np.load(moves_file)
    labels = np.load(labels_file)
    game_outcomes = np.load(game_outcomes_file)
    with open(move_to_idx_file, "r") as f:
        move_to_idx = json.load(f)
else:
    print("Preprocessing data from JSON file...")

    # Load the preprocessed JSON file
    with open(processed_json_path, "r") as file:
        processed_games = json.load(file)  # Load the JSON array

    # Prepare data for training
    fens = []
    moves = []
    labels = []
    game_outcomes = []

    for game in tqdm(processed_games, desc="Processing games"):
        outcome = game["game_outcome"]  # Game outcome (-1, 0, 1)
        for i in range(len(game["fens"]) - 1):  # Exclude the last FEN, as no move follows it
            fens.append(fen_to_tensor(game["fens"][i]))  # Current FEN
            moves.append(game["moves"][:i + 1])  # All moves leading up to the current FEN
            labels.append(game["moves"][i + 1])  # The next move after the current FEN
            game_outcomes.append(outcome)  # Add game outcome to each position

    fens = np.array(fens)
    # Convert game outcomes to appropriate format
    outcome_map = {-1: 0, 0: 1, 1: 2}
    game_outcomes = np.array([outcome_map[outcome] for outcome in game_outcomes]).reshape(-1, 1)  # Reshape for model input compatibility

    # Encode labels
    unique_moves = sorted(set(labels))
    move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
    labels = np.array([move_to_idx[label] for label in labels])

    # Encode and pad move sequences
    moves_encoded = [uci_to_tensor(seq, move_to_idx) for seq in moves]
    moves_padded = pad_sequences(moves_encoded, padding="post")  # Pad sequences to the same length

    moves = np.array(moves_padded)

    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save(fens_file, fens)
    np.save(moves_file, moves)
    np.save(labels_file, labels)
    np.save(game_outcomes_file, game_outcomes)
    with open(move_to_idx_file, "w") as f:
        json.dump(move_to_idx, f)

# Split data
X_fens_train, X_fens_test, X_moves_train, X_moves_test, X_outcomes_train, X_outcomes_test, y_train, y_test = train_test_split(
    fens, moves, game_outcomes, labels, test_size=0.2, random_state=42
)

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
    move_pred = Dense(64, activation="relu")(out2)
    move_pred = Dropout(0.3)(move_pred)
    next_move_output = Dense(num_moves, activation="softmax", name="next_move_output")(move_pred)

    # Final output for game outcome prediction
    outcome_pred = Dense(32, activation="relu")(out2)
    outcome_pred = Dense(3, activation="softmax", name="outcome_output")(outcome_pred)

    model = Model(inputs=[fen_input, move_input], outputs=[next_move_output, outcome_pred])
    return model

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    initial_lr = 1e-4
    decay_epochs = 10  # Total number of epochs
    alpha = 0.1  # Final learning rate as a fraction of the initial learning rate
    return initial_lr * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(np.pi * epoch / decay_epochs)))

input_fen_shape = (fens.shape[1],)
input_move_shape = (moves.shape[1],)
num_moves = len(move_to_idx)

model = create_transformer_model(input_fen_shape, input_move_shape, num_moves)

model.compile(
    optimizer="adam",
    loss={"next_move_output": "sparse_categorical_crossentropy", "outcome_output": "sparse_categorical_crossentropy"},
    metrics={"next_move_output": "accuracy", "outcome_output": "accuracy"}
)

# Callbacks
checkpoint = ModelCheckpoint("models/checkpoints3/model_checkpoint.h5", save_best_only=True, monitor="val_loss", mode="min")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model
print("Training model...")
history = model.fit(
    [X_fens_train, X_moves_train],
    {"next_move_output": y_train, "outcome_output": X_outcomes_train},
    validation_data=([X_fens_test, X_moves_test], {"next_move_output": y_test, "outcome_output": X_outcomes_test}),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stopping, lr_schedule]
)

# Save the final model
model.save("models/checkpoints3/model_final_with_outcome.h5")
print("Model training complete and saved.")
