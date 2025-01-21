"""
Chess Move and Game Outcome Prediction

This script implements a complete pipeline for training a Transformer-based deep learning model to predict chess moves and game outcomes based on FEN states and UCI move sequences.

Key Features:
1. **Data Preprocessing**:
   - Converts FEN strings and UCI moves into numeric tensor representations.
   - Maps game outcomes into discrete classes.
   - Pads move sequences to uniform lengths.

2. **Model Design**:
   - Utilizes a Transformer-based architecture for embedding FEN states and move sequences.
   - Employs dual outputs for next-move prediction (classification) and game outcome prediction (classification).

3. **Training Pipeline**:
   - Includes mechanisms for loading or preprocessing data.
   - Incorporates callbacks for checkpointing, early stopping, and learning rate scheduling.
   - Saves the trained model and intermediate preprocessing outputs for future use.

4. **Performance Metrics**:
   - Tracks and reports the following metrics during training:
     - Loss: Total training loss.
     - Next Move Loss: Cross-entropy loss for predicting the next chess move.
     - Outcome Loss: Cross-entropy loss for predicting the game outcome.
     - Next Move Accuracy: Accuracy of next move predictions.
     - Outcome Accuracy: Accuracy of game outcome predictions.

Usage:
1. Update the paths for the preprocessed JSON file and checkpoint directory.
2. Run the script to preprocess data, train the model, and save results.

Results:
- loss: 2.5458 - next_move_output_loss: 1.6852 - outcome_output_loss: 0.8606 - next_move_output_accuracy: 0.4622 - outcome_output_accuracy: 0.5014 - val_loss: 2.5144 - val_next_move_output_loss: 1.6496 - val_outcome_output_loss: 0.8648 - val_next_move_output_accuracy: 0.4698 - val_outcome_output_accuracy: 0.4936
"""
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

def fen_to_tensor(fen):
    """
    Converts a FEN string into a numeric tensor representation.

    Args:
        fen (str): FEN string representing the board state.

    Returns:
        np.ndarray: Tensor representation of the FEN.
    """
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
    """
    Converts UCI move sequences into their corresponding indices.

    Args:
        moves (list): List of UCI move strings.
        move_map (dict): Mapping of UCI moves to indices.

    Returns:
        list: List of indices representing the moves.
    """
    return [move_map[move] for move in moves if move in move_map]

def preprocess_data(json_path):
    """
    Preprocess data from the provided JSON file.

    Args:
        json_path (str): Path to the JSON file containing game data.

    Returns:
        tuple: Preprocessed tensors for FENs, moves, labels, game outcomes, and the move-to-index mapping.
    """
    with open(json_path, "r") as file:
        games = json.load(file)

    fens, moves, labels, outcomes = [], [], [], []
    for game in tqdm(games, desc="Processing games"):
        outcome = game["game_outcome"]  # Game outcome (-1, 0, 1)
        for i in range(len(game["fens"]) - 1):
            fens.append(fen_to_tensor(game["fens"][i]))
            moves.append(game["moves"][:i + 1])
            labels.append(game["moves"][i + 1])
            outcomes.append(outcome)

    outcome_map = {-1: 0, 0: 1, 1: 2}
    outcomes = np.array([outcome_map[outcome] for outcome in outcomes]).reshape(-1, 1)

    unique_moves = sorted(set(labels))
    move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
    print(f"Min index: {min(move_to_idx.values())}, Max index: {max(move_to_idx.values())}")
    print(f"Expected input_dim (total unique moves): {len(move_to_idx)}")
    labels = np.array([move_to_idx[label] for label in labels])
    
    moves_encoded = [uci_to_tensor(seq, move_to_idx) for seq in moves]
    moves_padded = pad_sequences(moves_encoded, padding="post")

    return np.array(fens), np.array(moves_padded), labels, outcomes, move_to_idx

def save_preprocessed_data(fens, moves, labels, outcomes, move_to_idx, output_dir):
    """
    Save preprocessed data to files.

    Args:
        fens (np.ndarray): FEN tensor data.
        moves (np.ndarray): Moves tensor data.
        labels (np.ndarray): Labels tensor data.
        outcomes (np.ndarray): Game outcomes tensor data.
        move_to_idx (dict): Move-to-index mapping.
        output_dir (str): Directory to save the files.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "fens.npy"), fens)
    np.save(os.path.join(output_dir, "moves.npy"), moves)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "game_outcomes.npy"), outcomes)
    with open(os.path.join(output_dir, "move_to_idx.json"), "w") as f:
        json.dump(move_to_idx, f)

def load_preprocessed_data(output_dir):
    """
    Load preprocessed data from files.

    Args:
        output_dir (str): Directory containing preprocessed files.

    Returns:
        tuple: Loaded tensors for FENs, moves, labels, game outcomes, and the move-to-index mapping.
    """
    fens = np.load(os.path.join(output_dir, "fens.npy"))
    moves = np.load(os.path.join(output_dir, "moves.npy"))
    labels = np.load(os.path.join(output_dir, "labels.npy"))
    outcomes = np.load(os.path.join(output_dir, "game_outcomes.npy"))
    with open(os.path.join(output_dir, "move_to_idx.json"), "r") as f:
        move_to_idx = json.load(f)
    return fens, moves, labels, outcomes, move_to_idx

def create_transformer_model(input_fen_shape, input_move_shape, num_moves):
    """
    Create a Transformer-based model for chess move and game outcome prediction.

    The model takes two inputs:
    1. FEN (Forsyth-Edwards Notation) tensor representing the chessboard state.
    2. Move sequence tensor representing previous moves in UCI (Universal Chess Interface) format.

    Outputs:
    1. `next_move_output`: Predicted next move as a probability distribution over all possible moves.
    2. `outcome_output`: Predicted game outcome (Win, Draw, Loss) as a probability distribution.

    Args:
        input_fen_shape (tuple): Shape of the FEN input tensor, e.g., (64,) or (8, 8, 13).
        input_move_shape (tuple): Shape of the move sequence input tensor, e.g., (max_sequence_length,).
        num_moves (int): Total number of unique possible moves.

    Returns:
        tensorflow.keras.Model: A compiled Transformer-based model for chess prediction tasks.
    """
    fen_input = Input(shape=input_fen_shape, name="fen_input")
    move_input = Input(shape=input_move_shape, name="move_input")

    fen_emb = Embedding(input_dim=12 * 8 + 2, output_dim=64)(fen_input)
    fen_emb = GlobalAveragePooling1D()(fen_emb)

    move_emb = Embedding(input_dim=num_moves, output_dim=64)(move_input)
    move_emb = GlobalAveragePooling1D()(move_emb)

    combined = Concatenate()([fen_emb, move_emb])
    x = Dense(128, activation="relu")(combined)

    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(tf.expand_dims(x, axis=1), tf.expand_dims(x, axis=1))
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + tf.squeeze(attn_output, axis=1))

    ffn = Dense(128, activation="relu")(out1)
    ffn = Dropout(0.1)(ffn)
    ffn_output = Dense(128)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    move_pred = Dense(64, activation="relu")(out2)
    move_pred = Dropout(0.3)(move_pred)
    next_move_output = Dense(num_moves, activation="softmax", name="next_move_output")(move_pred)

    outcome_pred = Dense(32, activation="relu")(out2)
    outcome_pred = Dense(3, activation="softmax", name="outcome_output")(outcome_pred)

    model = Model(inputs=[fen_input, move_input], outputs=[next_move_output, outcome_pred])
    return model

def lr_scheduler(epoch, lr):
    """
    Cosine decay learning rate scheduler.

    Adjusts the learning rate dynamically during training to improve convergence and prevent overfitting.

    Args:
        epoch (int): The current epoch number.
        lr (float): The current learning rate.

    Returns:
        float: Adjusted learning rate for the current epoch.

    Parameters:
    -----------
    - `initial_lr`: The starting learning rate (default is 1e-4).
    - `decay_epochs`: Total number of epochs over which decay occurs (default is 10).
    - `alpha`: The final learning rate as a fraction of the initial learning rate (default is 0.1).
    """
    initial_lr = 1e-4
    decay_epochs = 10
    alpha = 0.1
    return initial_lr * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(np.pi * epoch / decay_epochs)))

def train_model(PROCESSED_JSON_PATH, CHECKPOINT_DIR, FENS_FILE, MOVES_FILE, LABELS_FILE, GAME_OUTCOMES_FILE, MOVE_TO_IDX_FILE):
    """
    Train a Transformer-based model for chess move prediction and game outcome classification.

    This function handles the end-to-end process of training the model, including:
    1. Loading or preprocessing data from a JSON file.
    2. Preparing input tensors for FEN states, moves, and labels.
    3. Initializing and compiling a Transformer-based model.
    4. Training the model using callbacks for checkpointing, early stopping, and learning rate scheduling.
    5. Saving the trained model and relevant files.

    Args:
        PROCESSED_JSON_PATH (str): Path to the preprocessed JSON file containing game data.
        CHECKPOINT_DIR (str): Directory where preprocessed data and model checkpoints are stored.
        FENS_FILE (str): File path for storing/loading FEN tensor data.
        MOVES_FILE (str): File path for storing/loading move tensor data.
        LABELS_FILE (str): File path for storing/loading label tensor data.
        GAME_OUTCOMES_FILE (str): File path for storing/loading game outcome tensor data.
        MOVE_TO_IDX_FILE (str): File path for storing/loading the move-to-index mapping.

    Returns:
        None: Outputs are saved as files in the specified checkpoint directory.

    Notes:
        - Preprocessing includes converting FEN states and moves into numeric tensors.
        - The model is trained for both move prediction and outcome classification tasks.
        - Learning rate scheduling uses a cosine decay function for gradual adjustment.
    """
    if all(os.path.exists(file) for file in [FENS_FILE, MOVES_FILE, LABELS_FILE, GAME_OUTCOMES_FILE, MOVE_TO_IDX_FILE]):
        fens, moves, labels, outcomes, move_to_idx = load_preprocessed_data(CHECKPOINT_DIR)
    else:
        fens, moves, labels, outcomes, move_to_idx = preprocess_data(PROCESSED_JSON_PATH)
        save_preprocessed_data(fens, moves, labels, outcomes, move_to_idx, CHECKPOINT_DIR)

    X_fens_train, X_fens_test, X_moves_train, X_moves_test, X_outcomes_train, X_outcomes_test, y_train, y_test = train_test_split(
        fens, moves, outcomes, labels, test_size=0.2, random_state=42
    )

    input_fen_shape = (fens.shape[1],)
    input_move_shape = (moves.shape[1],)
    num_moves = len(move_to_idx)

    model = create_transformer_model(input_fen_shape, input_move_shape, num_moves)

    model.compile(
        optimizer="adam",
        loss={"next_move_output": "sparse_categorical_crossentropy", "outcome_output": "sparse_categorical_crossentropy"},
        metrics={"next_move_output": "accuracy", "outcome_output": "accuracy"}
    )

    checkpoint = ModelCheckpoint(
        os.path.join(CHECKPOINT_DIR, "model_checkpoint.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_schedule = LearningRateScheduler(lr_scheduler)

    print("Training model...")
    history = model.fit(
        [X_fens_train, X_moves_train],
        {"next_move_output": y_train, "outcome_output": X_outcomes_train},
        validation_data=([X_fens_test, X_moves_test], {"next_move_output": y_test, "outcome_output": X_outcomes_test}),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stopping, lr_schedule]
    )

    model.save(os.path.join(CHECKPOINT_DIR, "model_final_with_outcome.h5"))
    print("Model training complete and saved.")

if __name__ == "__main__":
    PROCESSED_JSON_PATH = r"D:\checkmate_ai\game_phases\open_data.json"
    CHECKPOINT_DIR = "../models/checkpoints"
    FENS_FILE = os.path.join(CHECKPOINT_DIR, "fens.npy")
    MOVES_FILE = os.path.join(CHECKPOINT_DIR, "moves.npy")
    LABELS_FILE = os.path.join(CHECKPOINT_DIR, "labels.npy")
    GAME_OUTCOMES_FILE = os.path.join(CHECKPOINT_DIR, "game_outcomes.npy")
    MOVE_TO_IDX_FILE = os.path.join(CHECKPOINT_DIR, "move_to_idx.json")

    train_model(PROCESSED_JSON_PATH, CHECKPOINT_DIR, FENS_FILE, MOVES_FILE, LABELS_FILE, GAME_OUTCOMES_FILE, MOVE_TO_IDX_FILE)
    

    