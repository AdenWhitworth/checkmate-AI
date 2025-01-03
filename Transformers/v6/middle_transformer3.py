#loss: 9.7301 - next_move_output_loss: 4.8537 - cp_outputs_loss: 0.0035 - mate_outputs_loss: 3.2657e-04 - next_move_output_accuracy: 0.1117 - next_move_output_top_5_accuracy: 0.0000e+00 - cp_outputs_mae: 0.0149 - mate_outputs_mae: 0.0075 - val_loss: 12.1662 - val_next_move_output_loss: 6.0719 - val_cp_outputs_loss: 0.0034 - val_mate_outputs_loss: 2.4817e-04 - val_next_move_output_accuracy: 0.0637 - val_next_move_output_top_5_accuracy: 0.0000e+00 - val_cp_outputs_mae: 0.0137 - val_mate_outputs_mae: 0.0063
#loss: 4.6842 - next_move_output_loss: 4.6769 - cp_outputs_loss: 0.0032 - mate_outputs_loss: 3.6313e-05 - next_move_output_accuracy: 0.1322 - next_move_output_<lambda>: 0.2988 - cp_outputs_mae: 0.0098 - mate_outputs_mae: 0.0013 - val_loss: 6.0562 - val_next_move_output_loss: 6.0489 - val_cp_outputs_loss: 0.0031 - val_mate_outputs_loss: 1.8966e-05 - val_next_move_output_accuracy: 0.0678 - val_next_move_output_<lambda>: 0.1779 - val_cp_outputs_mae: 0.0095 - val_mate_outputs_mae: 8.6648e-04
#700 games loss: 4.8450 - next_move_output_loss: 4.8378 - cp_outputs_loss: 0.0032 - mate_outputs_loss: 4.0608e-05 - next_move_output_accuracy: 0.1142 - next_move_output_top_k_accuracy: 0.2713 - cp_outputs_mae: 0.0101 - mate_outputs_mae: 0.0015 - val_loss: 6.0103 - val_next_move_output_loss: 6.0032 - val_cp_outputs_loss: 0.0031 - val_mate_outputs_loss: 2.0520e-05 - val_next_move_output_accuracy: 0.0713 - val_next_move_output_top_k_accuracy: 0.1823 - val_cp_outputs_mae: 0.0096 - val_mate_outputs_mae: 9.2591e-04
#2200 games loss: 4.2049 - next_move_output_loss: 4.1999 - cp_outputs_loss: 0.0031 - mate_outputs_loss: 1.5088e-05 - next_move_output_accuracy: 0.1594 - next_move_output_top_k_accuracy: 0.3713 - cp_outputs_mae: 0.0095 - mate_outputs_mae: 4.8922e-04 - val_loss: 5.0888 - val_next_move_output_loss: 5.0837 - val_cp_outputs_loss: 0.0031 - val_mate_outputs_loss: 6.3321e-06 - val_next_move_output_accuracy: 0.1087 - val_next_move_output_top_k_accuracy: 0.2786 - val_cp_outputs_mae: 0.0093 - val_mate_outputs_mae: 3.0456e-04
#3200 games loss: 4.1148 - next_move_output_loss: 4.1104 - cp_outputs_loss: 0.0030 - mate_outputs_loss: 8.9957e-06 - next_move_output_accuracy: 0.1583 - next_move_output_top_k_accuracy: 0.3817 - cp_outputs_mae: 0.0091 - mate_outputs_mae: 3.5462e-04 - val_loss: 4.6129 - val_next_move_output_loss: 4.6085 - val_cp_outputs_loss: 0.0030 - val_mate_outputs_loss: 5.7073e-06 - val_next_move_output_accuracy: 0.1324 - val_next_move_output_top_k_accuracy: 0.3217 - val_cp_outputs_mae: 0.0090 - val_mate_outputs_mae: 8.4046e-05
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

# File paths
processed_json_path = r"D:\checkmate_ai\game_phases\midgame_data3.jsonl"
fens_file = "models/checkpoints9/fens.npy"
moves_file = "models/checkpoints9/moves.npy"
labels_file = "models/checkpoints9/labels.npy"
cp_evaluations_file = "models/checkpoints9/cp_evaluations.npy"
mate_evaluations_file = "models/checkpoints9/mate_evaluations.npy"
move_to_idx_file = "models/checkpoints9/move_to_idx.json"

# FEN preprocessing functions
def fen_to_tensor_enhanced(fen):
    """
    Convert FEN to a tensor with spatial representation.
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
    return [move_map[move] for move in moves if move in move_map]

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

    # Step 1: Collect unique moves
    print("Collecting unique moves...")
    unique_moves = set()
    with open(processed_json_path, "r") as file:
        for line in tqdm(file, desc="Scanning games"):
            game = json.loads(line)
            unique_moves.update(game["legal_moves"])  # Collect all legal moves from data

    # Create move_to_idx mapping
    move_to_idx = {move: idx for idx, move in enumerate(sorted(unique_moves))}

    # Step 2: Preprocess data
    print("Processing games for data preprocessing...")
    with open(processed_json_path, "r") as file:
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

# Updated Model Architecture
def create_transformer_model(input_fen_shape, input_move_shape, num_moves):
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

# Learning rate scheduler
def lr_scheduler(epoch, lr):
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

# Initialize model
input_fen_shape = (8, 8, 13)  # Updated shape for enhanced FEN representation
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

checkpoint = ModelCheckpoint("models/checkpoints9/model_midgame_checkpoint.h5", save_best_only=True, monitor="val_loss", mode="min")
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
model.save("models/checkpoints9/model_midgame_final.h5")
print("Model training complete and saved.")
