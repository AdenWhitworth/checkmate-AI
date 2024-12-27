#Checkpoint 3: loss: 5.6935 - policy_output_loss: 5.6935 - value_output_loss: 1.2366e-05 - policy_output_accuracy: 0.0421 - policy_output_top_3_accuracy: 0.1025 - value_output_mae: 0.0023 - val_loss: 5.7074 - val_policy_output_loss: 5.7074 - val_value_output_loss: 1.1315e-05 - val_policy_output_accuracy: 0.0418 - val_policy_output_top_3_accuracy: 0.1010 - val_value_output_mae: 0.0023
#Checkpoint 4: loss: 5.5582 - policy_output_loss: 5.5582 - value_output_loss: 1.2631e-05 - policy_output_accuracy: 0.0474 - policy_output_top_3_accuracy: 0.1138 - value_output_mae: 0.0025
import os
import numpy as np
import tensorflow as tf
import chess
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt

def encode_board(fen):
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board = chess.Board(fen)
    board_tensor = np.zeros((8, 8, 16), dtype=np.float32)  # Adjust for 16 channels

    # Encode pieces
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        board_tensor[row, col, piece_to_index[piece.symbol()]] = 1.0

    # Encode turn
    board_tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Encode castling rights
    castling = (
        (board.has_kingside_castling_rights(chess.WHITE) << 3)
        | (board.has_queenside_castling_rights(chess.WHITE) << 2)
        | (board.has_kingside_castling_rights(chess.BLACK) << 1)
        | (board.has_queenside_castling_rights(chess.BLACK))
    )
    board_tensor[:, :, 13] = castling / 15.0

    # Encode en passant square
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        board_tensor[row, col, 14] = 1.0

    return board_tensor

# Create move index map
def create_move_index_map():
    move_indices = []
    for src in range(64):
        for dst in range(64):
            promotion = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            for promo in promotion:
                move = chess.Move(src, dst, promotion=promo)
                move_indices.append(move.uci())
    return {uci: idx for idx, uci in enumerate(move_indices)}

def prepare_data_with_history(jsonl_path, move_index_map, max_eval=500, history_length=5):
    X, y_policy, y_value = [], [], []
    valid_labels = set(move_index_map.values())  # Precompute valid labels

    with open(jsonl_path) as file:
        for line in tqdm(file, desc="Processing data entries", unit="entries"):
            entry = json.loads(line)

            # Extract data
            fen = entry.get("fen")
            next_move = entry.get("next_move")
            value = entry.get("value")
            move_history = entry.get("move_history", [])[-history_length:]

            if not fen or not next_move or value is None:
                print(f"Skipping entry due to missing fields: {entry}")
                continue  # Skip incomplete entries

            board = chess.Board(fen)
            if chess.Move.from_uci(next_move) not in board.legal_moves:
                print(f"Skipping entry due to missing fields: {entry}")
                continue  # Skip if the move is not legal

            # Encode the board and history
            encoded_board = encode_board(fen)
            try:
                history_encoded = [move_index_map[move] for move in move_history]
            except KeyError:
                print(f"Skipping entry due to missing fields: {entry}")
                continue  # Skip if any move in history is not in the map

            # Pad the move history to `history_length`
            if len(history_encoded) < history_length:
                history_encoded = [0] * (history_length - len(history_encoded)) + history_encoded

            # Encode the next move
            move_index = move_index_map.get(next_move)
            if move_index is None or move_index not in valid_labels:
                continue  # Skip invalid moves

            # Append data
            X.append((encoded_board, history_encoded))
            y_policy.append(move_index)
            y_value.append((value + max_eval) / (2 * max_eval))  # Normalize value

    # Convert to NumPy arrays
    board_states = np.array([x[0] for x in X], dtype=np.float32)
    move_histories = np.array([x[1] for x in X], dtype=np.int32)  # Now padded and consistent
    y_policy = np.array(y_policy, dtype=np.int64)  # Ensure it's 1D
    y_value = np.array(y_value, dtype=np.float32)

    return board_states, move_histories, y_policy, y_value

def data_generator_with_history(board_states, move_histories, y_policy, y_value, batch_size):
    final_batch_size = len(board_states) % batch_size
    if final_batch_size:
        print(f"Final batch size: {final_batch_size}")

    while True:
        for i in range(0, len(board_states), batch_size):
            board_batch = board_states[i:i + batch_size]
            history_batch = move_histories[i:i + batch_size]
            y_policy_batch = y_policy[i:i + batch_size]
            y_value_batch = y_value[i:i + batch_size]

            yield {"board_input": board_batch, "history_input": history_batch}, {"policy_output": y_policy_batch, "value_output": y_value_batch}

def create_transformer_model_with_history(num_moves):
    board_input = tf.keras.Input(shape=(8, 8, 16), name="board_input")
    x_board = tf.keras.layers.Reshape((64, 16))(board_input)

    history_input = tf.keras.Input(shape=(None,), name="history_input")
    x_history = tf.keras.layers.Embedding(input_dim=num_moves, output_dim=64, mask_zero=True)(history_input)

    board_length = tf.shape(x_board)[1]
    history_length = tf.shape(x_history)[1]
    max_length = tf.maximum(board_length, history_length)

    # Padding for x_board
    x_board_padded = tf.pad(x_board, [[0, 0], [0, max_length - board_length], [0, 0]])

    # Project `x_board_padded` to match the feature dimension of `x_history_padded`
    x_board_projected = tf.keras.layers.Dense(64)(x_board_padded)

    # Padding for x_history
    padding_needed = max_length - history_length
    padding_tensor = tf.zeros_like(x_history[:, :padding_needed, :])
    x_history_padded = tf.concat([x_history, padding_tensor], axis=1)

    x_combined = tf.keras.layers.Concatenate(axis=1)([x_board_projected, x_history_padded])
    x_combined = tf.keras.layers.Masking()(x_combined)

    # Transformer layers
    for _ in range(6):
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(x_combined, x_combined)
        x_combined = tf.keras.layers.LayerNormalization()(x_combined + attn_output)
        ff_output = tf.keras.layers.Dense(64, activation="relu")(x_combined)
        x_combined = tf.keras.layers.LayerNormalization()(x_combined + ff_output)

    # Pool the combined representation
    pooled_representation = tf.keras.layers.GlobalAveragePooling1D()(x_combined)

    # Outputs
    policy_output = tf.keras.layers.Dense(num_moves, activation="softmax", name="policy_output")(pooled_representation)
    value_output = tf.keras.layers.Dense(1, activation="linear", name="value_output")(pooled_representation)

    return tf.keras.Model(inputs=[board_input, history_input], outputs=[policy_output, value_output])

def lr_scheduler(epoch, lr):
    initial_lr = 1e-4
    decay_epochs = 10  # Total number of epochs
    alpha = 0.1  # Final learning rate as a fraction of the initial learning rate
    return initial_lr * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(np.pi * epoch / decay_epochs)))

def top_3_accuracy(y_true, y_pred):
    # Ensure y_true is 1D by reshaping
    y_true = tf.squeeze(y_true, axis=-1) if len(y_true.shape) > 1 else y_true
    
    # Convert to int32 if not already integers
    if y_true.dtype != tf.int32 and y_true.dtype != tf.int64:
        y_true = tf.cast(y_true, tf.int32)
    
    # Convert y_true to one-hot encoding
    num_classes = tf.shape(y_pred)[-1]  # Determine the number of classes dynamically
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    
    return tf.keras.metrics.top_k_categorical_accuracy(y_true=y_true_one_hot, y_pred=y_pred, k=3)

# Training
def train_model(model, train_gen, val_gen, train_steps_per_epoch, val_steps_per_epoch, epochs=10):
    checkpoint_dir = "models/checkpoints4"
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "model_checkpoint.h5"), save_best_only=True, monitor="val_loss", mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=1),
        LearningRateScheduler(lr_scheduler, verbose=1)
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss={"policy_output": "sparse_categorical_crossentropy", "value_output": "mse"},
        loss_weights={"policy_output": 1.0, "value_output": 0.2},
        metrics={"policy_output": ["accuracy", top_3_accuracy], "value_output": ["mae"]},
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks
    )

# Main Script
jsonl_path = r"D:\checkmate_ai\evaluated_training_data2.jsonl"
data_file = "models/checkpoints4/processed_data.npz"
batch_size = 32
move_index_map_file = "models/checkpoints4/id_to_move.json"

# Create or load the move index map
if os.path.exists(move_index_map_file):
    with open(move_index_map_file, "r") as f:
        move_index_map = json.load(f)
else:
    move_index_map = create_move_index_map()
    with open(move_index_map_file, "w") as f:
        json.dump(move_index_map, f)

# Load or prepare data
if os.path.exists(data_file):
    print(f"Loading processed data from {data_file}")
    data = np.load(data_file)
    board_states, move_histories, y_policy, y_value = data["board_states"], data["move_histories"], data["y_policy"], data["y_value"]
    print("Keys in .npz file:", data.files)
else:
    print(f"{data_file} not found. Preparing data from {jsonl_path}")
    board_states, move_histories, y_policy, y_value = prepare_data_with_history(jsonl_path, move_index_map)
    np.savez_compressed(data_file, board_states=board_states, move_histories=move_histories, y_policy=y_policy, y_value=y_value)

# Split data into training and validation
split_index = int(len(board_states) * 0.8)
X_train_boards, X_val_boards = board_states[:split_index], board_states[split_index:]
X_train_history, X_val_history = move_histories[:split_index], move_histories[split_index:]
y_policy_train, y_policy_val = y_policy[:split_index], y_policy[split_index:]
y_value_train, y_value_val = y_value[:split_index], y_value[split_index:]

# Create data generators
train_gen = data_generator_with_history(X_train_boards, X_train_history, y_policy_train, y_value_train, batch_size)
val_gen = data_generator_with_history(X_val_boards, X_val_history, y_policy_val, y_value_val, batch_size)

# Calculate steps
train_steps_per_epoch = max(1, len(X_train_boards) // batch_size)
val_steps_per_epoch = max(1, len(X_val_boards) // batch_size)

# Create the model
num_moves = len(move_index_map)
model = create_transformer_model_with_history(num_moves)

# Train the model
history = train_model(model, train_gen, val_gen, train_steps_per_epoch, val_steps_per_epoch, epochs=10)

# Plotting Training Results
def plot_training_results(history):
    metrics = {
        "Policy Loss": ["policy_output_loss", "val_policy_output_loss"],
        "Value Loss": ["value_output_loss", "val_value_output_loss"],
        "Policy Accuracy": ["policy_output_accuracy", "val_policy_output_accuracy"],
        "Top-3 Accuracy": ["policy_output_top_3_accuracy", "val_policy_output_top_3_accuracy"],
        "Value MAE": ["value_output_mae", "val_value_output_mae"],
    }
    
    plt.figure(figsize=(15, 12))
    for idx, (title, keys) in enumerate(metrics.items(), 1):
        train_key, val_key = keys
        train_data = history.history.get(train_key, [])
        val_data = history.history.get(val_key, [])
        
        plt.subplot(3, 2, idx)
        plt.plot(train_data, label=f"Train {title}")
        plt.plot(val_data, label=f"Val {title}")
        plt.xlabel("Epochs")
        plt.ylabel(title)
        plt.title(f"{title} Over Epochs")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Call the function
plot_training_results(history)
