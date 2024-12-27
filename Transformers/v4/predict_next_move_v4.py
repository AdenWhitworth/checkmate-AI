import chess
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# Load the trained model
model_path = "models/checkpoints/model_checkpoint.h5"
model = tf.keras.models.load_model(model_path)

# Load the move index map
with open("models/checkpoints/id_to_move.json", "r") as f:
    move_index_map = json.load(f)
id_to_move = {v: k for k, v in move_index_map.items()}

# Encode the board state
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

# Validate UCI move
def is_valid_uci_move(move):
    try:
        chess.Move.from_uci(move)
        return True
    except chess.InvalidMoveError:
        return False

# Predict the next move
def predict_next_move(fen, policy_weight=0.7, value_weight=0.3):
    """
    Predict the next move based on policy output and board evaluation (value output).
    
    Args:
    - fen (str): The board position in FEN notation.
    - policy_weight (float): Weight assigned to the policy output in the score calculation.
    - value_weight (float): Weight assigned to the value output in the score calculation.
    
    Returns:
    - str: The best move in UCI format.
    - list: Top legal moves with their scores.
    - float: Board evaluation value.
    """
    board = chess.Board(fen)
    encoded_board = encode_board(fen)
    encoded_board = np.expand_dims(encoded_board, axis=0)  # Add batch dimension

    # Get predictions
    policy_output, value_output = model.predict(encoded_board)
    value_output = value_output[0][0]  # Extract scalar value

    # Sort policy predictions
    sorted_indices = np.argsort(policy_output[0])[::-1]
    sorted_moves = [id_to_move[idx] for idx in sorted_indices if idx in id_to_move]
    sorted_scores = policy_output[0][sorted_indices]

    # Filter and score legal moves
    legal_moves = []
    for move, score in zip(sorted_moves, sorted_scores):
        if is_valid_uci_move(move) and chess.Move.from_uci(move) in board.legal_moves:
            combined_score = policy_weight * score + value_weight * value_output
            legal_moves.append((move, combined_score))

    # Sort legal moves by combined score
    legal_moves.sort(key=lambda x: x[1], reverse=True)

    # Select the best move
    best_move = legal_moves[0][0] if legal_moves else None

    return best_move, legal_moves, value_output

# Example usage
if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Initial chess position
    best_move, legal_moves, position_value = predict_next_move(fen)

    print(f"Best move: {best_move}")
    print("Top legal moves with scores:")
    for move, score in legal_moves[:5]:  # Show top 5 moves
        print(f"Move: {move}, Score: {score:.4f}")
    print(f"Position value: {position_value:.4f}")
