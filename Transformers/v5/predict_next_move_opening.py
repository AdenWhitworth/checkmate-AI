import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import chess

# Load preprocessed mappings and model
move_to_idx_file = "models/checkpoints/move_to_idx.json"
model_file = "models/checkpoints/model_final_updated.h5"

with open(move_to_idx_file, "r") as f:
    move_to_idx = json.load(f)

# Reverse the move_to_idx mapping for decoding
idx_to_move = {v: k for k, v in move_to_idx.items()}

# Load the trained model
model = load_model(model_file)

# Preprocessing functions
def fen_to_tensor(fen):
    """
    Convert FEN string to tensor representation.
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
    Convert UCI move sequence to tensor representation.
    """
    return [move_map[move] for move in moves if move in move_map]

def is_legal_move(fen, move):
    """
    Check if a move is legal in the given FEN position.
    """
    board = chess.Board(fen)
    return chess.Move.from_uci(move) in board.legal_moves

# Function to predict the next move
def predict_next_move(fen, moves, model, max_move_length=28):
    """
    Predict the next best move given a FEN and move history.
    """
    # Prepare FEN tensor
    fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)

    # Prepare moves tensor with padding
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Make prediction
    prediction = model.predict([fen_tensor, moves_tensor])
    sorted_indices = np.argsort(prediction[0])[::-1]  # Sort by confidence

    # Find the first legal move
    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]
        if is_legal_move(fen, predicted_move):
            return predicted_move
    return None  # No legal move found

# Example usage
fen = "rnbqkb1r/pp2pppp/3p4/8/3nP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 5"  # Example FEN
moves = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]  # Example move sequence

predicted_move = predict_next_move(fen, moves, model)
print(f"Predicted next move: {predicted_move}")
