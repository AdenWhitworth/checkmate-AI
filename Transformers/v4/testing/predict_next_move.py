"""
Script to Predict the Next Move in a Chess Game Using a Trained TensorFlow Model

Overview:
This script leverages a pre-trained TensorFlow model to predict the next move in a chess game. It processes the current board state and move history into the required input format for the model, ensuring that predictions are both valid and legal according to the chess rules.

Key Features:
1. Loads a trained TensorFlow model and move-to-index mappings from JSON files.
2. Encodes the current board state (in FEN format) and move history into a format suitable for model inference.
3. Predicts the next move while validating its legality on the provided board state.
4. Supports configurable weights for policy (move prediction) and value (board evaluation) outputs to fine-tune prediction scoring.

Functions:
- `load_mappings`: Loads the move-to-index and index-to-move mappings from JSON files.
- `encode_board`: Encodes the current board state (FEN) into a tensor format required by the model.
- `encode_move_history`: Converts a move history into a fixed-length sequence for model input.
- `is_valid_uci_move`: Validates whether a UCI move string is well-formed and exists in the move mapping.
- `predict_next_move`: Generates the best move prediction based on model output and board state legality.

Usage Instructions:
1. Update the file paths for the trained model and JSON mappings (`model_path` and `move_index_map`).
2. Provide the board state (FEN) and move history as input for predictions.
3. Execute the script to obtain the best predicted move and the top legal moves with their scores.
"""

import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

def top_3_accuracy(y_true, y_pred):
    """
    Calculate top-3 accuracy for policy predictions.

    Args:
        y_true (tf.Tensor): True move indices (ground truth labels).
        y_pred (tf.Tensor): Predicted probabilities for all moves.

    Returns:
        tf.Tensor: Top-3 categorical accuracy metric.
    """
    # Ensure y_true is 1D by reshaping
    y_true = tf.squeeze(y_true, axis=-1) if len(y_true.shape) > 1 else y_true
    
    # Convert to int32 if not already integers
    if y_true.dtype != tf.int32 and y_true.dtype != tf.int64:
        y_true = tf.cast(y_true, tf.int32)
    
    # Convert y_true to one-hot encoding
    num_classes = tf.shape(y_pred)[-1]  # Determine the number of classes dynamically
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    
    return tf.keras.metrics.top_k_categorical_accuracy(y_true=y_true_one_hot, y_pred=y_pred, k=3)

def encode_board(fen):
    """
    Encode a chess board state from FEN into a 3D tensor representation.

    Args:
        fen (str): FEN string representing the board state.

    Returns:
        np.ndarray: Encoded 8x8x16 tensor representation of the board:
            - Channels 0-11: Piece positions (white and black pieces).
            - Channel 12: Turn indicator (1.0 for white, 0.0 for black).
            - Channel 13: Castling rights (encoded as a normalized integer).
            - Channel 14: En passant square (1.0 for the target square, 0.0 otherwise).
    """
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board = chess.Board(fen)
    board_tensor = np.zeros((8, 8, 16), dtype=np.float32)  # Adjust for 16 channels

    # Encode pieces
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        board_tensor[row, col, piece_to_index[piece.symbol()]] = 1.0

    # Encode turn
    board_tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Encode castling rights
    castling = (
        (board.has_kingside_castling_rights(chess.WHITE) << 3)
        | (board.has_queenside_castling_rights(chess.WHITE) << 2)
        | (board.has_kingside_castling_rights(chess.BLACK) << 1)
        | (board.has_queenside_castling_rights(chess.BLACK))
    )
    board_tensor[:, :, 13] = castling / 15.0

    # Encode en passant square
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        board_tensor[row, col, 14] = 1.0

    return board_tensor

def encode_move_history(moves, move_index_map, history_length=5):
    """
    Encode a move history into a fixed-length sequence of integers.

    Args:
        moves (list): List of UCI moves in the game's history.
        move_index_map (dict): Mapping of UCI moves to indices.
        history_length (int): Maximum number of moves to include.

    Returns:
        np.ndarray: Encoded move history, padded to `history_length`.
    """
    history_encoded = [move_index_map.get(move, 0) for move in moves[-history_length:]]
    if len(history_encoded) < history_length:
        history_encoded = [0] * (history_length - len(history_encoded)) + history_encoded
    return np.array(history_encoded, dtype=np.int32)

def is_valid_uci_move(move, board):
    """
    Validate a UCI move string against the current board state.

    Args:
        move (str): The UCI move string to validate.
        board (chess.Board): The current board state.

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    try:
        uci_move = chess.Move.from_uci(move)
        return uci_move in board.legal_moves
    except ValueError:
        return False

def predict_next_move(fen, moves, model_path, id_to_move_path, policy_weight=0.7, value_weight=0.3, history_length=5):
    """
    Predict the next move based on policy output and board evaluation (value output).

    Args:
        fen (str): The board position in FEN notation.
        moves (list): List of UCI moves in the game's history.
        policy_weight (float): Weight assigned to the policy output in the score calculation.
        value_weight (float): Weight assigned to the value output in the score calculation.
        history_length (int): Maximum number of moves to include from history.

    Returns:
        str: The best move in UCI format.
        list: Top legal moves with their scores.
        float: Board evaluation value (as predicted by the model).
    """

    custom_objects = {"top_3_accuracy": top_3_accuracy}
    model = load_model(model_path, custom_objects=custom_objects)

    # Load the move index map
    with open(id_to_move_path, "r") as f:
        move_index_map = json.load(f)
    id_to_move = {v: k for k, v in move_index_map.items()}

    board = chess.Board(fen)
    encoded_board = encode_board(fen)
    encoded_history = encode_move_history(moves, move_index_map, history_length)

    # Add batch dimensions
    encoded_board = np.expand_dims(encoded_board, axis=0)
    encoded_history = np.expand_dims(encoded_history, axis=0)

    # Normalize weights
    total_weight = policy_weight + value_weight
    policy_weight /= total_weight
    value_weight /= total_weight

    # Get predictions
    policy_output, value_output = model.predict([encoded_board, encoded_history], verbose=0)
    value_output = value_output[0][0]  # Extract scalar value

    # Get sorted policy predictions
    sorted_indices = np.argsort(policy_output[0])[::-1]  # Descending order
    sorted_moves = [id_to_move[idx] for idx in sorted_indices if idx in id_to_move]
    sorted_scores = policy_output[0][sorted_indices]

    # Filter and score legal moves
    legal_moves = []
    for move, score in zip(sorted_moves, sorted_scores):
        if not is_valid_uci_move(move, board):
            continue
        uci_move = chess.Move.from_uci(move)
        combined_score = policy_weight * score + value_weight * value_output
        legal_moves.append((move, combined_score))

    # Sort legal moves by combined score
    legal_moves.sort(key=lambda x: x[1], reverse=True)

    # Select the best move
    best_move = legal_moves[0][0] if legal_moves else None

    return best_move, legal_moves, value_output

# Example usage
if __name__ == "__main__":
    fen = "r2q1rk1/pp1nbppp/3p1n2/2p1p3/2P1P3/2N1BN2/PPQ2PPP/R3K2R w KQ - 0 10"
    moves = ["e2e4", "e7e5", "g1f3", "d7d6", "c2c4", "g8f6", "b1c3", "f8e7", "f1e2", "e8g8", "e1g1"]

    model_path = "../models/checkpoints4/model_checkpoint.h5"
    id_to_move_path = "../models/checkpoints4/id_to_move.json"

    best_move, legal_moves, position_value = predict_next_move(fen, moves, model_path, id_to_move_path)

    print(f"Best move: {best_move}")
    print("Top legal moves with scores:")
    for move, score in legal_moves[:5]:  # Show top 5 moves
        print(f"Move: {move}, Score: {score:.4f}")
    print(f"Position value: {position_value:.4f}")
