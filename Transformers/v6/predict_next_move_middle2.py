import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import chess

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

# Load preprocessed mappings and model
move_to_idx_file = "models/checkpoints7/move_to_idx.json"
model_file = "models/checkpoints7/model_midgame_final.h5"

with open(move_to_idx_file, "r") as f:
    move_to_idx = json.load(f)

# Reverse the move_to_idx mapping for decoding
idx_to_move = {v: k for k, v in move_to_idx.items()}

custom_objects = {"top_k_accuracy": top_k_accuracy}
model = load_model(model_file, custom_objects=custom_objects)

# Preprocessing functions
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
    """
    Convert UCI move sequence to tensor representation.
    """
    return [move_map[move] for move in moves if move in move_map]

def is_legal_move(fen, move):
    """
    Check if a move is legal in the given FEN position.
    """
    board = chess.Board(fen)
    return chess.Move.from_uci(move) in board.legal_moves

def predict_top_moves_and_eval(fen, moves, model, max_move_length=161, top_n=5):
    """
    Predict the top N moves and their evaluations from a given FEN and move history.
    """
    fen_tensor = np.expand_dims(fen_to_tensor_enhanced(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Predict using the model
    move_pred, cp_preds, mate_preds = model.predict([fen_tensor, moves_tensor])

    # Sort moves by predicted probabilities
    sorted_indices = np.argsort(move_pred[0])[::-1]

    # Create a chess.Board for legality checks
    board = chess.Board(fen)

    top_moves = []
    top_pred_moves = []

    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]

        top_pred_moves.append({
            "move": predicted_move,
            "probability": move_pred[0][idx],
            "cp_eval": cp_preds[0][idx] * 1000.0,  # Scale back to centipawns
            "mate_eval": mate_preds[0][idx]
        })

        # Check if the move is legal
        if chess.Move.from_uci(predicted_move) in board.legal_moves:
            top_moves.append({
                "move": predicted_move,
                "probability": move_pred[0][idx],
                "cp_eval": cp_preds[0][idx] * 1000.0,  # Scale back to centipawns
                "mate_eval": mate_preds[0][idx]
            })
            if len(top_moves) == top_n:
                break

    return top_moves, top_pred_moves

# Example FEN and move sequence
fen = "r2q1rk1/pp1nbppp/3p1n2/2p1p3/2P1P3/2N1BN2/PPQ2PPP/R3K2R w KQ - 0 10"
#moves = ["e2e4", "e7e5", "g1f3", "d7d6", "c2c4", "g8f6", "b1c3", "f8e7", "f1e2", "e8g8", "e1g1"]
moves = ["e1g1"]

top_moves, top_pred_moves = predict_top_moves_and_eval(fen, moves, model, top_n=5)

# Print the top N moves and their evaluations
print("Top (legal) Moves and Evaluations:")
for i, move_data in enumerate(top_moves, start=1):
    print(f"Rank {i}: Move: {move_data['move']}, Probability: {move_data['probability']:.4f}, "
          f"CP Evaluation: {move_data['cp_eval']:.2f} (centipawns), Mate Evaluation: {move_data['mate_eval']:.2f}")
    
# Print the top N moves and their evaluations
print("Top (legal/illegal) Moves and Evaluations:")
for i, move_data in enumerate(top_pred_moves, start=1):
    print(f"Rank {i}: Move: {move_data['move']}, Probability: {move_data['probability']:.4f}, "
          f"CP Evaluation: {move_data['cp_eval']:.2f} (centipawns), Mate Evaluation: {move_data['mate_eval']:.2f}")

