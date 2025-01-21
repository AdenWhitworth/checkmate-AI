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
import onnxruntime as ort

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

def fen_to_tensor_onnx(fen, fixed_length=65):
    """
    Convert FEN string to a numeric tensor representation within the embedding range,
    ensuring the tensor has a fixed length.

    Args:
        fen (str): FEN string representing the board state.
        fixed_length (int): Fixed length for the FEN tensor (default: 65).

    Returns:
        np.ndarray: Tensor representation of the FEN with fixed length.
    """
    char_to_index = {
        'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,  # Black pieces
        'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12,  # White pieces
        '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,  # Empty squares
        '/': 0  # Separator (optional for some models)
    }

    board, turn, _, _, _, _ = fen.split()
    board_tensor = []
    for char in board:
        if char in char_to_index:
            board_tensor.append(char_to_index[char])
        else:
            raise ValueError(f"Unexpected FEN character: {char}")

    turn_tensor = [1] if turn == 'w' else [0]

    full_tensor = board_tensor + turn_tensor

    if len(full_tensor) < fixed_length:
        full_tensor.extend([0] * (fixed_length - len(full_tensor)))
    elif len(full_tensor) > fixed_length:
        full_tensor = full_tensor[:fixed_length]

    return np.array(full_tensor, dtype=np.int32)

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

    fen_tensor = np.expand_dims(fen_to_tensor(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    move_pred, outcome_pred = model.predict([fen_tensor, moves_tensor])

    sorted_indices = np.argsort(move_pred[0])[::-1]
    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]
        if is_legal_move(fen, predicted_move):
            break

    reverse_outcome_map = {0: "Loss", 1: "Draw", 2: "Win"}
    predicted_outcome = reverse_outcome_map[np.argmax(outcome_pred[0])]

    return predicted_move, predicted_outcome

def predict_next_move_and_outcome_onnx(fen, moves, CHECKPOINT_DIR, max_move_length=28):
    """
    Predict the next move and game outcome for a given chess position and move history using ONNX model.
    """
    with open(os.path.join(CHECKPOINT_DIR, "move_to_idx.json"), "r") as f:
        move_to_idx = json.load(f)
    idx_to_move = {v: k for k, v in move_to_idx.items()}

    onnx_session = ort.InferenceSession(os.path.join(CHECKPOINT_DIR, "onnx_model/model_final_with_outcome.onnx"))

    fen_tensor = np.expand_dims(fen_to_tensor_onnx(fen), axis=0).astype(np.float32)

    move_indices = uci_to_tensor(moves, move_to_idx)
    if len(move_indices) < max_move_length:
        move_indices += [0] * (max_move_length - len(move_indices)) 
    else:
        move_indices = move_indices[:max_move_length]

    moves_tensor = np.expand_dims(np.array(move_indices, dtype=np.float32), axis=0)

    onnx_inputs = {"args_0": fen_tensor, "args_1": moves_tensor}
    onnx_outputs = onnx_session.run(None, onnx_inputs)
    move_pred = onnx_outputs[0]
    outcome_pred = onnx_outputs[1]

    sorted_indices = np.argsort(move_pred[0])[::-1]
    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]
        if is_legal_move(fen, predicted_move):
            break

    reverse_outcome_map = {0: "Loss", 1: "Draw", 2: "Win"}
    predicted_outcome = reverse_outcome_map[np.argmax(outcome_pred[0])]

    return predicted_move, predicted_outcome


if __name__ == "__main__":
    CHECKPOINT_DIR = "../models/checkpoints"

    fen = "rnbqkb1r/pp2pppp/3p4/8/3nP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 5"
    moves = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]
    
    
    predicted_move, predicted_outcome = predict_next_move_and_outcome(fen, moves, CHECKPOINT_DIR)
    print(f"Predicted next move: {predicted_move}")
    print(f"Predicted game outcome: {predicted_outcome}")


    CHECKPOINT_DIR = "../models/checkpoints"
    predicted_move, predicted_outcome = predict_next_move_and_outcome_onnx(fen, moves, CHECKPOINT_DIR)
    print(f"Onnx Predicted next move: {predicted_move}")
    print(f"Onnx Predicted game outcome: {predicted_outcome}")
