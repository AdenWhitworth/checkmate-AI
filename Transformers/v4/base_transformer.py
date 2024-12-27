#loss: 5.2088 - policy_output_loss: 5.0924 - value_output_loss: 0.0627 - policy_output_accuracy: 0.1211 - value_output_mae: 0.1989 - val_loss: 5.1112 - val_policy_output_loss: 5.0028 - val_value_output_loss: 0.0103 - val_policy_output_accuracy: 0.1314 - val_value_output_mae: 0.0806
import os
import numpy as np
import tensorflow as tf
import chess
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

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

# Step 2: Prepare the dataset
def prepare_data(jsonl_path, move_index_map, max_eval=500):
    X, y_policy, y_value = [], [], []

    with open(jsonl_path) as file:
        for line in tqdm(file, desc="Processing data entries", unit="entries"):
            entry = json.loads(line)
            fen = entry["fen"]
            next_move = entry["next_move"]
            value = entry["value"]

            # Create and validate the board
            board = chess.Board(fen)
            if chess.Move.from_uci(next_move) not in board.legal_moves:
                continue

            # Encode the board state
            encoded_board = encode_board(fen)

            # Map the next move
            move_index = move_index_map.get(next_move)
            if move_index is None:
                continue

            X.append(encoded_board)
            y_policy.append(move_index)
            y_value.append((value + max_eval) / (2 * max_eval))  # Normalize to [0, 1]

    # Encode policy labels
    le = LabelEncoder()
    y_policy = le.fit_transform(y_policy)

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y_value = np.array(y_value, dtype=np.float32)

    return X, y_policy, y_value, le

# Step 3: Data generator for batching
def data_generator(X, y_policy, y_value, batch_size):
    while True:  # Infinite generator
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_policy_batch = y_policy[i:i + batch_size]
            y_value_batch = y_value[i:i + batch_size]

            yield X_batch, {"policy_output": y_policy_batch, "value_output": y_value_batch}

# Step 4: Define the Transformer model
def create_transformer_model(num_moves):
    board_input = tf.keras.Input(shape=(8, 8, 14), name="board_input")
    x = tf.keras.layers.Reshape((64, 14))(board_input)

    for _ in range(2):  # Reduced from 4 to 2 layers
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=14)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_output)
        ff_output = tf.keras.layers.Dense(
            14,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            kernel_initializer="lecun_normal",
        )(x)
        x = tf.keras.layers.LayerNormalization()(x + ff_output)
        x = tf.keras.layers.Dropout(0.1)(x)

    shared_representation = tf.keras.layers.Flatten()(x)
    policy_output = tf.keras.layers.Dense(num_moves, activation='softmax', name="policy_output")(shared_representation)
    value_output = tf.keras.layers.Dense(1, activation="linear", name="value_output")(shared_representation)
    return tf.keras.Model(inputs=board_input, outputs=[policy_output, value_output])

# Step 5: Train the model
def train_model(model, train_gen, val_gen, train_steps_per_epoch, val_steps_per_epoch, epochs=10):
    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_checkpoint.h5"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=1)
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0),
        loss={"policy_output": "sparse_categorical_crossentropy", "value_output": "mse"},
        loss_weights={"policy_output": 1.0, "value_output": 0.1},
        metrics={"policy_output": "accuracy", "value_output": "mae"}
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks
    )

# Step 6: Create move index map
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
data_file = "models/checkpoints/processed_data.npz"
batch_size = 32
move_index_map_file = "models/checkpoints/id_to_move.json"

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
    X, y_policy, y_value = data["X"], data["y_policy"], data["y_value"]
    print("Keys in .npz file:", data.files)
else:
    print(f"{data_file} not found. Preparing data from {jsonl_path}")
    X, y_policy, y_value, le = prepare_data(jsonl_path, move_index_map)
    np.savez_compressed(data_file, X=X, y_policy=y_policy, y_value=y_value)

# Split data into training and validation
split_index = int(len(X) * 0.8)
X_train, X_val = X[:split_index], X[split_index:]
y_policy_train, y_policy_val = y_policy[:split_index], y_policy[split_index:]
y_value_train, y_value_val = y_value[:split_index], y_value[split_index:]

# Create data generators
train_gen = data_generator(X_train, y_policy_train, y_value_train, batch_size)
val_gen = data_generator(X_val, y_policy_val, y_value_val, batch_size)

# Calculate steps
train_steps_per_epoch = len(X_train) // batch_size
val_steps_per_epoch = len(X_val) // batch_size

# Create the model
num_moves = len(move_index_map)
model = create_transformer_model(num_moves)

# Train the model
train_model(model, train_gen, val_gen, train_steps_per_epoch, val_steps_per_epoch, epochs=10)

