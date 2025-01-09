"""
Chess Midgame Data Preprocessing and Model Training Script
===========================================================
This script preprocesses chess midgame data from a JSONL file and trains a deep learning model using 
a Transformer-based architecture. The model predicts the next move, centipawn (CP) evaluations, and 
mate evaluations for a given chess board state and sequence of moves.

Features:
---------
1. **FEN to Tensor Conversion**: Converts FEN strings into spatial tensors with enhanced representation.
2. **Data Preprocessing**: Parses JSONL files to extract FENs, moves, and evaluation data.
3. **Transformer Model Architecture**: Combines CNN for board representation and attention mechanisms 
   for move sequences.
4. **Loss and Metrics**: Utilizes custom loss functions and top-k accuracy metrics for better training insights.
5. **Learning Rate Scheduling**: Employs cosine decay for smoother convergence.
6. **Model Checkpoints**: Saves the best model weights based on validation loss.

Inputs:
-------
- JSONL file containing processed chess games.
- Pretrained model checkpoints (if available).

Outputs:
--------
- Trained model saved in HDF5 format.
- Preprocessed tensors for FENs, moves, labels (grand master moves), CP evaluations, and mate evaluations.
- Log files summarizing training and validation performance.

Results:
- Checkpoints 7 (700 games): loss: 4.8450 - next_move_output_loss: 4.8378 - cp_outputs_loss: 0.0032 - mate_outputs_loss: 4.0608e-05 - next_move_output_accuracy: 0.1142 - next_move_output_top_k_accuracy: 0.2713 - cp_outputs_mae: 0.0101 - mate_outputs_mae: 0.0015 - val_loss: 6.0103 - val_next_move_output_loss: 6.0032 - val_cp_outputs_loss: 0.0031 - val_mate_outputs_loss: 2.0520e-05 - val_next_move_output_accuracy: 0.0713 - val_next_move_output_top_k_accuracy: 0.1823 - val_cp_outputs_mae: 0.0096 - val_mate_outputs_mae: 9.2591e-04
- Checkpoints 8 (2200 games): loss: 4.2049 - next_move_output_loss: 4.1999 - cp_outputs_loss: 0.0031 - mate_outputs_loss: 1.5088e-05 - next_move_output_accuracy: 0.1594 - next_move_output_top_k_accuracy: 0.3713 - cp_outputs_mae: 0.0095 - mate_outputs_mae: 4.8922e-04 - val_loss: 5.0888 - val_next_move_output_loss: 5.0837 - val_cp_outputs_loss: 0.0031 - val_mate_outputs_loss: 6.3321e-06 - val_next_move_output_accuracy: 0.1087 - val_next_move_output_top_k_accuracy: 0.2786 - val_cp_outputs_mae: 0.0093 - val_mate_outputs_mae: 3.0456e-04
- Checkpoints 9 (3200 games): loss: 4.1148 - next_move_output_loss: 4.1104 - cp_outputs_loss: 0.0030 - mate_outputs_loss: 8.9957e-06 - next_move_output_accuracy: 0.1583 - next_move_output_top_k_accuracy: 0.3817 - cp_outputs_mae: 0.0091 - mate_outputs_mae: 3.5462e-04 - val_loss: 4.6129 - val_next_move_output_loss: 4.6085 - val_cp_outputs_loss: 0.0030 - val_mate_outputs_loss: 5.7073e-06 - val_next_move_output_accuracy: 0.1324 - val_next_move_output_top_k_accuracy: 0.3217 - val_cp_outputs_mae: 0.0090 - val_mate_outputs_mae: 8.4046e-05

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, GlobalAveragePooling1D, Concatenate, Conv2D, Flatten, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import json
import os

# FEN preprocessing functions
def fen_to_tensor_enhanced(fen):
    """
    Convert FEN to a tensor representation with spatial encoding.

    Args:
        fen (str): FEN string representing the chess board state.

    Returns:
        np.ndarray: Tensor representation of the FEN with shape (8, 8, 13).
    """
    piece_map = {'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,
                 'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12}
    tensor = np.zeros((8, 8, 12), dtype=np.int32)
    board, turn = fen.split()[:2]

    row, col = 0, 0
    for char in board:
        if char.isdigit():
            col += int(char)
        elif char == '/':
            row += 1
            col = 0
        else:
            piece_idx = piece_map[char]
            tensor[row, col, piece_idx - 1] = 1
            col += 1

    turn_tensor = np.ones((8, 8, 1)) if turn == 'w' else np.zeros((8, 8, 1))
    return np.concatenate([tensor, turn_tensor], axis=-1)

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

def load_preprocessed_data(output_dir):
    """
    Load preprocessed data from files.

    Args:
        output_dir (str): Directory containing preprocessed files.

    Returns:
        tuple: Loaded tensors for FENs, moves, labels, cp_evaluations, mate_evaluations, and the move-to-index mapping.
    """
    fens = np.load(os.path.join(output_dir, "fens.npy"))
    moves = np.load(os.path.join(output_dir, "moves.npy"))
    labels = np.load(os.path.join(output_dir, "labels.npy"))
    cp_evaluations = np.load(os.path.join(output_dir, "cp_evaluations.npy"))
    mate_evaluations = np.load(os.path.join(output_dir, "mate_evaluations.npy"))
    with open(os.path.join(output_dir, "move_to_idx.json"), "r") as f:
        move_to_idx = json.load(f)
    return fens, moves, labels, cp_evaluations, mate_evaluations, move_to_idx

def preprocess_data(json_path):
    """
    Preprocess data from the provided JSON file.

    Args:
        json_path (str): Path to the JSON file containing game data.

    Returns:
        tuple: Preprocessed tensors for FENs, moves, labels, cp_evaluations, mate_evaluations, and the move-to-index mapping.
    """
    print("Preprocessing data from JSONL file...")
    fens = []
    moves = []
    labels = []
    cp_evaluations = []
    mate_evaluations = []

    # Step 1: Collect unique moves
    print("Collecting unique moves...")
    unique_moves = set()
    with open(json_path, "r") as file:
        for line in tqdm(file, desc="Scanning games"):
            game = json.loads(line)
            unique_moves.update(game["legal_moves"])  # Collect all legal moves from data

    # Create move_to_idx mapping
    move_to_idx = {move: idx for idx, move in enumerate(sorted(unique_moves))}

    # Step 2: Preprocess data
    print("Processing games for data preprocessing...")
    with open(json_path, "r") as file:
        for line in tqdm(file, desc="Processing games"):
            game = json.loads(line)
            fens.append(fen_to_tensor_enhanced(game["fen"]))  # FEN before the move
            moves.append(game["moves"])             # All moves leading up to the current position
            labels.append(game["next_move"])        # The next move made by the grandmaster

            # Initialize evaluations for all possible moves
            cp_eval_for_moves = [0.0] * len(move_to_idx)
            mate_eval_for_moves = [0.0] * len(move_to_idx)

            for move, cp_eval, mate_eval in zip(game["legal_moves"], game["cp_evals"], game["mate_evals"]):
                if move in move_to_idx:
                    idx = move_to_idx[move]
                    cp_eval_for_moves[idx] = cp_eval / 1000.0  # Normalize CP to [-1, 1]
                    mate_eval_for_moves[idx] = mate_eval       # Keep Mate evaluations as-is

            cp_evaluations.append(cp_eval_for_moves)
            mate_evaluations.append(mate_eval_for_moves)

    # Convert to numpy arrays
    fens = np.array(fens, dtype=np.int32)
    cp_evaluations = np.array(cp_evaluations, dtype=np.float32)
    mate_evaluations = np.array(mate_evaluations, dtype=np.float32)

    # Encode labels
    labels = np.array([move_to_idx[label] for label in labels])

    # Encode and pad move sequences
    moves_encoded = [uci_to_tensor(seq, move_to_idx) for seq in moves]
    moves_padded = pad_sequences(moves_encoded, padding="post")
    moves = np.array(moves_padded)

    return fens, moves, labels, cp_evaluations, mate_evaluations, move_to_idx

def save_preprocessed_data(fens, moves, labels, cp_evaluations, mate_evaluations, move_to_idx, output_dir):
    """
    Save preprocessed data to files.

    Args:
        fens (np.ndarray): FEN tensor data.
        moves (np.ndarray): Moves tensor data.
        labels (np.ndarray): Labels tensor data.
        cp_evaluations (np.ndarray): CP evals tensor data.
        mate_evaluations (np.ndarray): Mate evals tensor data.
        move_to_idx (dict): Move-to-index mapping.
        output_dir (str): Directory to save the files.
    """
    print("Saving preprocessed data...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "fens.npy"), fens)
    np.save(os.path.join(output_dir, "moves.npy"), moves)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "cp_evaluations.npy"), cp_evaluations)
    np.save(os.path.join(output_dir, "mate_evaluations.npy"), mate_evaluations)
    with open(os.path.join(output_dir, "move_to_idx.json"), "w") as f:
        json.dump(move_to_idx, f)

def create_transformer_model(input_fen_shape, input_move_shape, num_moves):
    """
    Create a Transformer-based model for chess move prediction and evaluation.

    This model is designed for multi-task learning:
    1. Predicting the next move in a chess game.
    2. Evaluating the quality of legal moves in terms of centipawn (CP) scores and mate possibilities.

    Args:
        input_fen_shape (tuple): Shape of the FEN (Forsyth-Edwards Notation) input tensor, typically (8, 8, 13) for spatial board encoding.
        input_move_shape (tuple): Shape of the move sequence input tensor, e.g., (max_sequence_length,).
        num_moves (int): Total number of unique possible moves.

    Returns:
        tensorflow.keras.Model: A compiled Transformer-based model for chess tasks.
    """
    fen_input = Input(shape=input_fen_shape, name="fen_input")
    move_input = Input(shape=input_move_shape, name="move_input")

    # CNN for FEN (spatial relationships)
    fen_cnn = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(fen_input)
    fen_cnn = Flatten()(fen_cnn)
    fen_emb = Dense(128, activation="relu")(fen_cnn)

    # Custom Attention for move sequences
    move_emb = Embedding(input_dim=num_moves, output_dim=64)(move_input)
    query = Dense(64)(move_emb)
    key = Dense(64)(move_emb)
    value = Dense(64)(move_emb)
    move_attention = MultiHeadAttention(num_heads=4, key_dim=64)(query, key, value)
    move_emb = tf.reduce_mean(move_attention, axis=1)  # Aggregate over sequence

    # Combine embeddings
    combined = Concatenate()([fen_emb, move_emb])

    # Transformer block
    x = Dense(128, activation="relu")(combined)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(tf.expand_dims(x, axis=1), tf.expand_dims(x, axis=1))
    attn_output = Dropout(0.2)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + tf.squeeze(attn_output, axis=1))

    ffn = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(out1)
    ffn = Dropout(0.2)(ffn)
    ffn_output = Dense(128)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Final output for next move prediction
    x = Dense(64, activation="relu")(out2)
    x = Dropout(0.3)(x)
    next_move_output = Dense(num_moves, activation="softmax", name="next_move_output")(x)

    # Outputs for CP and Mate evaluations for all moves
    cp_outputs = Dense(64, activation="relu")(out2)
    cp_outputs = Dense(num_moves, activation="linear", name="cp_outputs")(cp_outputs)

    mate_outputs = Dense(64, activation="relu")(out2)
    mate_outputs = Dense(num_moves, activation="linear", name="mate_outputs")(mate_outputs)

    model = Model(inputs=[fen_input, move_input], outputs=[next_move_output, cp_outputs, mate_outputs])
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

def top_k_accuracy(y_true, y_pred, k=5):
    """
    Top-k categorical accuracy for sparse labels.
    Args:
        y_true: Ground truth labels (integer class indices).
        y_pred: Predicted probabilities.
        k: The number of top predictions to consider.

    Returns:
        Top-k categorical accuracy as a scalar tensor.
    """
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

def train_model(PROCESSED_JSON_PATH, CHECKPOINT_DIR, FENS_FILE, MOVES_FILE, LABELS_FILE, CP_EVALUATIONS_FILE, MATE_EVALUATAIONS_FILE, MOVE_TO_IDX_FILE):
    """
    Train a Transformer-based model for chess move prediction and evaluation.

    This function handles the entire training pipeline, including:
    1. Loading or preprocessing data.
    2. Preparing the input tensors for FEN states, move sequences, centipawn evaluations, mate evaluations, and labels.
    3. Initializing and compiling a Transformer-based deep learning model.
    4. Training the model using callbacks for checkpointing, early stopping, and learning rate scheduling.
    5. Saving the trained model and its checkpoints for future use.

    Args:
        PROCESSED_JSON_PATH (str): Path to the JSON file containing preprocessed game data.
        CHECKPOINT_DIR (str): Directory to store model checkpoints and preprocessed data.
        FENS_FILE (str): Path to the file containing FEN tensor data.
        MOVES_FILE (str): Path to the file containing move sequence tensor data.
        LABELS_FILE (str): Path to the file containing label tensor data (next move indices).
        CP_EVALUATIONS_FILE (str): Path to the file containing centipawn (CP) evaluation tensor data.
        MATE_EVALUATAIONS_FILE (str): Path to the file containing mate evaluation tensor data.
        MOVE_TO_IDX_FILE (str): Path to the file containing the move-to-index mapping.

    Returns:
        None: Outputs are saved in the specified checkpoint directory.
    """

    if all(os.path.exists(file) for file in [FENS_FILE, MOVES_FILE, LABELS_FILE, CP_EVALUATIONS_FILE, MATE_EVALUATAIONS_FILE, MOVE_TO_IDX_FILE]):
        fens, moves, labels, cp_evaluations, mate_evaluations, move_to_idx = load_preprocessed_data(CHECKPOINT_DIR)
    else:
        fens, moves, labels, cp_evaluations, mate_evaluations, move_to_idx = preprocess_data(PROCESSED_JSON_PATH)
        save_preprocessed_data(fens, moves, labels, cp_evaluations, mate_evaluations, move_to_idx, CHECKPOINT_DIR)

    # Split data
    X_fens_train, X_fens_test, X_moves_train, X_moves_test, X_cp_train, X_cp_test, X_mate_train, X_mate_test, y_train, y_test = train_test_split(
        fens, moves, cp_evaluations, mate_evaluations, labels, test_size=0.2, random_state=42
    )

    # Initialize model
    input_fen_shape = (8, 8, 13)
    input_move_shape = (moves.shape[1],)
    num_moves = len(move_to_idx)

    model = create_transformer_model(input_fen_shape, input_move_shape, num_moves)

    # Modify Loss and Training
    model.compile(
        optimizer="adam",
        loss={
            "next_move_output": "sparse_categorical_crossentropy",
            "cp_outputs": "mean_squared_error",
            "mate_outputs": "mean_squared_error",
        },
        loss_weights={
            "next_move_output": 1.0,
            "cp_outputs": 0.01,
            "mate_outputs": 0.01,
        },
        metrics={
            "next_move_output": [
                "accuracy",
                top_k_accuracy,  # Direct reference to the standalone function
            ],
            "cp_outputs": "mae",
            "mate_outputs": "mae",
        },
    )

    checkpoint = ModelCheckpoint(
        os.path.join(CHECKPOINT_DIR, "model_checkpoint.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_schedule = LearningRateScheduler(lr_scheduler)

    # Train and evaluate the updated model
    history = model.fit(
        [X_fens_train, X_moves_train],
        {"next_move_output": y_train, "cp_outputs": X_cp_train, "mate_outputs": X_mate_train},
        validation_data=([X_fens_test, X_moves_test], {"next_move_output": y_test, "cp_outputs": X_cp_test, "mate_outputs": X_mate_test}),
        epochs=50,
        batch_size=64,
        callbacks=[checkpoint, early_stopping, lr_schedule],
    )

    # Save the final model
    model.save(os.path.join(CHECKPOINT_DIR, "model_midgame_final.h5"))
    print("Model training complete and saved.")

if __name__ == "__main__":
    PROCESSED_JSON_PATH = r"D:\checkmate_ai\game_phases\midgame_data3.jsonl"
    CHECKPOINT_DIR = "models/checkpoints9"
    FENS_FILE = os.path.join(CHECKPOINT_DIR, "fens.npy")
    MOVES_FILE = os.path.join(CHECKPOINT_DIR, "moves.npy")
    LABELS_FILE = os.path.join(CHECKPOINT_DIR, "labels.npy")
    CP_EVALUATIONS_FILE = os.path.join(CHECKPOINT_DIR, "cp_evaluations.npy")
    MATE_EVALUATAIONS_FILE = os.path.join(CHECKPOINT_DIR, "mate_evaluations.npy")
    MOVE_TO_IDX_FILE = os.path.join(CHECKPOINT_DIR, "move_to_idx.json")

    train_model(PROCESSED_JSON_PATH, CHECKPOINT_DIR, FENS_FILE, MOVES_FILE, LABELS_FILE, CP_EVALUATIONS_FILE, MATE_EVALUATAIONS_FILE, MOVE_TO_IDX_FILE)
    
