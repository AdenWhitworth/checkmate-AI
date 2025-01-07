"""
Transformer-Based Chess Move Prediction Model

This script:
1. Encodes chess board states from FEN into 3D tensor representations.
2. Prepares datasets for training from JSONL files containing FEN, next moves, and evaluation values.
3. Creates a Transformer-based neural network for policy and value prediction.
4. Trains the model with policy and value losses and evaluates its performance.
5. Saves the trained model, move mappings, and processed datasets.

Stockfish Preprocessing:
- Fen: Current chessboard state in FEN representation.
- Next Move: The next move made by the player in the game
- Value: CP & Mate evaluation from stockfish of the move made by the player
Example: {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "next_move": "g1f3", "value": 0.14}

Metrics:
- Loss: Combined policy and value loss during training.
- Policy Output Loss: Cross-entropy loss for predicting the next move.
- Value Output Loss: Mean squared error (MSE) for predicting board evaluation values.
- Policy Output Accuracy: Accuracy of move predictions.
- Value Output Mean Absolute Error (MAE): MAE for board evaluation predictions.

Results:
- Checkpoints: loss: 5.2088 - policy_output_loss: 5.0924 - value_output_loss: 0.0627 - policy_output_accuracy: 0.1211 - value_output_mae: 0.1989 - val_loss: 5.1112 - val_policy_output_loss: 5.0028 - val_value_output_loss: 0.0103 - val_policy_output_accuracy: 0.1314 - val_value_output_mae: 0.0806
"""
import os
import numpy as np
import tensorflow as tf
import chess
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def encode_board(fen):
    """
    Encode a chess board state from FEN into a 3D tensor representation.

    Args:
        fen (str): FEN string representing the board state.

    Returns:
        np.ndarray: Encoded 8x8x14 tensor representation of the board.
            - Channels 0-11: Piece locations (e.g., white pawn in channel 0, black king in channel 11).
            - Channel 12: Side to move (1.0 for white, 0.0 for black).
            - Channel 13: Castling rights (encoded as a normalized integer).
    """
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

def prepare_data(jsonl_path, move_index_map, max_eval=500):
    """
    Prepare the dataset for training from JSONL files.

    Args:
        jsonl_path (str): Path to the JSONL file with training data.
        move_index_map (dict): Mapping of moves to unique indices.
        max_eval (int): Maximum evaluation value for normalizing game scores.

    Returns:
        tuple:
            - np.ndarray: Encoded board states as a 4D tensor (N x 8 x 8 x 14).
            - np.ndarray: Encoded policy labels (indices of moves).
            - np.ndarray: Encoded value labels (normalized evaluation values in [0, 1]).
            - LabelEncoder: Fitted LabelEncoder for policy labels.
    """
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

def data_generator(X, y_policy, y_value, batch_size):
    """
    Generate batches of data for training and validation.

    Args:
        X (np.ndarray): Input board states (4D tensor).
        y_policy (np.ndarray): Policy labels (indices of moves).
        y_value (np.ndarray): Value labels (normalized evaluation values).
        batch_size (int): Number of samples per batch.

    Yields:
        tuple:
            - np.ndarray: Batch of input board states.
            - dict: Dictionary with:
                - "policy_output": Batch of policy labels.
                - "value_output": Batch of value labels.
    """
    while True:  # Infinite generator
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_policy_batch = y_policy[i:i + batch_size]
            y_value_batch = y_value[i:i + batch_size]

            yield X_batch, {"policy_output": y_policy_batch, "value_output": y_value_batch}

def create_transformer_model(num_moves):
    """
    Create a Transformer-based model for chess move prediction.

    Args:
        num_moves (int): Number of possible moves (output size for policy head).

    Returns:
        tf.keras.Model: Compiled Transformer model with two outputs:
            - `policy_output`: Softmax output for next-move prediction.
            - `value_output`: Linear output for evaluation value prediction.
    """
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

def train_model(jsonl_path, output_dir, epochs=10):
    """
    Train the Transformer-based model for chess move and evaluation prediction.

    Args:
        jsonl_path (str): Path to the JSONL file with training data.
        output_dir (str): Directory to save the trained model, checkpoints, and mappings.
        epochs (int): Number of epochs to train the model.

    Workflow:
        1. Creates or loads the move-to-index mapping.
        2. Prepares or loads the dataset (board states, policy labels, value labels).
        3. Splits the dataset into training and validation sets.
        4. Creates data generators for batch processing.
        5. Builds and compiles the Transformer-based model.
        6. Trains the model with early stopping and checkpoint saving.

    Saves:
        - Trained model checkpoint (HDF5 format).
        - Processed dataset (NPZ format).
        - Move-to-index mapping (JSON format).
    """
    move_index_map_file = os.path.join(output_dir, "id_to_move.json")
    if os.path.exists(move_index_map_file):
        with open(move_index_map_file, "r") as f:
            move_index_map = json.load(f)
    else:
        move_index_map = create_move_index_map()
        with open(move_index_map_file, "w") as f:
            json.dump(move_index_map, f)

    # Prepare or load data
    data_file = os.path.join(output_dir, "processed_data.npz")
    if os.path.exists(data_file):
        print(f"Loading processed data from {data_file}")
        data = np.load(data_file)
        X, y_policy, y_value = data["X"], data["y_policy"], data["y_value"]
    else:
        print(f"Preparing data from {jsonl_path}")
        X, y_policy, y_value, _ = prepare_data(jsonl_path, move_index_map)
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

    os.makedirs(output_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "model_checkpoint.h5"),
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

def create_move_index_map():
    """
    Create a mapping of UCI chess moves to unique indices.

    Returns:
        dict: Mapping of UCI move strings to unique integer indices.
            - Includes all possible moves, including promotions.
    """
    move_indices = []
    for src in range(64):
        for dst in range(64):
            promotion = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            for promo in promotion:
                move = chess.Move(src, dst, promotion=promo)
                move_indices.append(move.uci())
    return {uci: idx for idx, uci in enumerate(move_indices)}

if __name__ == "__main__":
    jsonl_path = r"D:\checkmate_ai\evaluated_training_data2.jsonl"
    output_dir = "../models/checkpoints"
    batch_size = 32

    train_model(jsonl_path, output_dir, epochs=10)

