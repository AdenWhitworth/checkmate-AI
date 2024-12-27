import os
import numpy as np
import tensorflow as tf
import chess
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tqdm import tqdm
import matplotlib.pyplot as plt

# Step 1: Encode the board state
def encode_board(fen):
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board = chess.Board(fen)
    board_tensor = np.zeros((8, 8, 14), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        board_tensor[row, col, piece_to_index[piece.symbol()]] = 1.0

    board_tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    castling = (
        (board.has_kingside_castling_rights(chess.WHITE) << 3)
        | (board.has_queenside_castling_rights(chess.WHITE) << 2)
        | (board.has_kingside_castling_rights(chess.BLACK) << 1)
        | (board.has_queenside_castling_rights(chess.BLACK))
    )
    board_tensor[:, :, 13] = castling / 15.0
    return board_tensor

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
                continue  # Skip incomplete entries

            board = chess.Board(fen)
            if chess.Move.from_uci(next_move) not in board.legal_moves:
                continue  # Skip if the move is not legal

            # Encode the board and history
            encoded_board = encode_board(fen)
            try:
                history_encoded = [move_index_map[move] for move in move_history]
            except KeyError:
                continue  # Skip if any move in history is not in the map

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
    move_histories = np.array([x[1] for x in X], dtype=np.int32)
    y_policy = np.array(y_policy, dtype=np.int64)  # Ensure it's 1D
    y_value = np.array(y_value, dtype=np.float32)

    return board_states, move_histories, y_policy, y_value

def data_generator_with_history(board_states, move_histories, y_policy, y_value, batch_size):
    while True:
        for i in range(0, len(board_states), batch_size):
            board_batch = board_states[i:i + batch_size]
            history_batch = move_histories[i:i + batch_size]
            y_policy_batch = y_policy[i:i + batch_size].astype(np.int64)  # Ensure 1D
            y_value_batch = y_value[i:i + batch_size]

            # Debugging generator outputs
            print("Generator: y_policy_batch shape:", y_policy_batch.shape)
            print("Generator: y_value_batch shape:", y_value_batch.shape)
            print("Generator: y_policy_batch values:", y_policy_batch[:10])  # Print first 10 elements
            print("Generator: y_value_batch values:", y_value_batch[:10])    # Print first 10 elements

            yield {"board_input": board_batch, "history_input": history_batch}, {"policy_output": y_policy_batch, "value_output": y_value_batch}

# Transformer model
def create_transformer_model_with_history(num_moves):
    board_input = tf.keras.Input(shape=(8, 8, 14), name="board_input")
    x_board = tf.keras.layers.Reshape((64, 14))(board_input)

    history_input = tf.keras.Input(shape=(None,), name="history_input")
    x_history = tf.keras.layers.Embedding(input_dim=num_moves, output_dim=14, mask_zero=True)(history_input)

    board_length = tf.shape(x_board)[1]
    history_length = tf.shape(x_history)[1]
    max_length = tf.maximum(board_length, history_length)

    # Padding for x_board
    x_board_padded = tf.pad(x_board, [[0, 0], [0, max_length - board_length], [0, 0]])

    # Padding for x_history
    padding_needed = max_length - history_length
    padding_tensor = tf.zeros_like(x_board_padded[:, :padding_needed, :])  # Zero-padding tensor
    x_history_padded = tf.concat([x_history, padding_tensor], axis=1)

    x_combined = tf.keras.layers.Concatenate(axis=1)([x_board_padded, x_history_padded])
    x_combined = tf.keras.layers.Masking()(x_combined)

    # Transformer layers
    for _ in range(2):
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=14)(x_combined, x_combined)
        x_combined = tf.keras.layers.LayerNormalization()(x_combined + attn_output)
        ff_output = tf.keras.layers.Dense(14, activation="relu")(x_combined)
        x_combined = tf.keras.layers.LayerNormalization()(x_combined + ff_output)

    # Pool the combined representation
    pooled_representation = tf.keras.layers.GlobalAveragePooling1D()(x_combined)

    # Outputs
    policy_output = tf.keras.layers.Dense(num_moves, activation="softmax", name="policy_output")(pooled_representation)
    value_output = tf.keras.layers.Dense(1, activation="linear", name="value_output")(pooled_representation)

    return tf.keras.Model(inputs=[board_input, history_input], outputs=[policy_output, value_output])

def top_3_accuracy(y_true, y_pred):
    """
    Custom top-3 accuracy metric with TensorFlow tf.print for debugging.
    """
    y_true = tf.reshape(y_true, [-1])  # Flatten to 1D tensor

    # Debugging with tf.print (compatible with graph execution)
    tf.print("Debug: y_true shape:", tf.shape(y_true), "y_pred shape:", tf.shape(y_pred))
    tf.print("Debug: y_true dtype:", y_true.dtype, "y_pred dtype:", y_pred.dtype)

    # Calculate top-k accuracy
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

# Training
def train_model(model, train_gen, val_gen, train_steps_per_epoch, val_steps_per_epoch, epochs=10):
    checkpoint_dir = "models/checkpoints3"
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "model_checkpoint.h5"), save_best_only=True, monitor="val_loss", mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=1)
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0),
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

# Main Script
jsonl_path = r"D:\checkmate_ai\evaluated_training_data2.jsonl"
data_file = "models/checkpoints3/processed_data.npz"
batch_size = 32
move_index_map_file = "models/checkpoints3/id_to_move.json"

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
train_steps_per_epoch = len(X_train_boards) // batch_size
val_steps_per_epoch = len(X_val_boards) // batch_size

# Create the model
num_moves = len(move_index_map)
model = create_transformer_model_with_history(num_moves)

# Train the model
history = train_model(model, train_gen, val_gen, train_steps_per_epoch, val_steps_per_epoch, epochs=10)

# Step 7: Visualization of Training Results
# Plot Policy Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['policy_output_loss'], label='Train Policy Loss')
plt.plot(history.history['val_policy_output_loss'], label='Val Policy Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Policy Output Loss Over Epochs')
plt.legend()
plt.show()

# Plot Value Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['value_output_loss'], label='Train Value Loss')
plt.plot(history.history['val_value_output_loss'], label='Val Value Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Value Output Loss Over Epochs')
plt.legend()
plt.show()

# Plot Policy Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['policy_output_accuracy'], label='Train Policy Accuracy')
plt.plot(history.history['val_policy_output_accuracy'], label='Val Policy Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Policy Output Accuracy Over Epochs')
plt.legend()
plt.show()

# Plot Value MAE
plt.figure(figsize=(10, 6))
plt.plot(history.history['value_output_mae'], label='Train Value MAE')
plt.plot(history.history['val_value_output_mae'], label='Val Value MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Value Output MAE Over Epochs')
plt.legend()
plt.show()

# Plot Top-k Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['policy_output_top_k_categorical_accuracy'], label='Train Top-5 Accuracy')
plt.plot(history.history['val_policy_output_top_k_categorical_accuracy'], label='Val Top-5 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Top-5 Policy Output Accuracy Over Epochs')
plt.legend()
plt.show()


