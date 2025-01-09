"""
Chess Move and Outcome Predictor

This script uses a trained TensorFlow model to predict the next move and the game's outcome based on a given FEN position and move history.

Features:
1. Converts FEN strings and UCI move sequences into numeric tensor representations.
2. Predicts the next legal move and the game's outcome using the trained model.
3. Validates predictions against the current board state to ensure legality.

Usage:
1. Provide a FEN position and a sequence of UCI moves leading to it.
2. Load the trained model and mappings.
3. Use `predict_next_move_and_outcome` to get predictions.

"""
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import chess
import os

def load_model_and_mappings(move_to_idx_file, model_file):
    """
    Load the trained model and mappings.

    Args:
        move_to_idx_file (str): Path to the move-to-index JSON file.
        model_file (str): Path to the trained model file.

    Returns:
        tuple: Loaded model, move-to-index mapping, and index-to-move mapping.
    """
    with open(move_to_idx_file, "r") as f:
        move_to_idx = json.load(f)
    idx_to_move = {v: k for k, v in move_to_idx.items()}
    model = load_model(model_file)
    return model, move_to_idx, idx_to_move

def fen_to_tensor(fen):
    """
    Convert FEN string to a numeric tensor representation.

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
    Convert a UCI move sequence to tensor representation.

    Args:
        moves (list): List of UCI move strings.
        move_map (dict): Mapping of UCI moves to indices.

    Returns:
        list: List of indices representing the moves.
    """
    return [move_map[move] for move in moves if move in move_map]

def is_legal_move(fen, move):
    """
    Check if a move is legal in the given FEN position.

    Args:
        fen (str): FEN string of the board state.
        move (str): UCI move string.

    Returns:
        bool: True if the move is legal, False otherwise.
    """
    board = chess.Board(fen)
    return chess.Move.from_uci(move) in board.legal_moves

def predict_next_move_and_outcome(fen, moves, CHECKPOINT_DIR, max_move_length=28):
    """
    Predict the next move and game outcome for a given chess position and move history.

    Args:
        fen (str): FEN string representing the current board state.
        moves (list): List of UCI moves leading up to the current board state.
        CHECKPOINT_DIR (str): Path to the model checkpoint directory.
        max_move_length (int, optional): Maximum length for move sequences. Defaults to 28.

    Returns:
        tuple:
            - predicted_move (str): The next move in UCI format.
            - predicted_outcome (str): The predicted game outcome as one of ["Loss", "Draw", "Win"].
    """
    model, move_to_idx, idx_to_move = load_model_and_mappings(os.path.join(CHECKPOINT_DIR, "move_to_idx.json"), os.path.join(CHECKPOINT_DIR, "model_final_with_outcome.h5"))

    # Prepare tensors
    fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Make predictions
    move_pred, outcome_pred = model.predict([fen_tensor, moves_tensor])

    # Decode next move
    sorted_indices = np.argsort(move_pred[0])[::-1]
    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]
        if is_legal_move(fen, predicted_move):
            break

    # Decode outcome prediction
    reverse_outcome_map = {0: "Loss", 1: "Draw", 2: "Win"}
    predicted_outcome = reverse_outcome_map[np.argmax(outcome_pred[0])]

    return predicted_move, predicted_outcome

if __name__ == "__main__":
    CHECKPOINT_DIR = "../models/checkpoints"

    # Example FEN and move sequence
    fen = "rnbqkb1r/pp2pppp/3p4/8/3nP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 5"
    moves = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]

    # Predict next move and outcome
    predicted_move, predicted_outcome = predict_next_move_and_outcome(fen, moves, CHECKPOINT_DIR)
    print(f"Predicted next move: {predicted_move}")
    print(f"Predicted game outcome: {predicted_outcome}")
