from stockfish import Stockfish
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import chess
import random
import json
import chess.pgn
import numpy as np

# Stockfish Skill Levels vs Human ELO (Approximation)
# -----------------------------------------------
# | Stockfish Level | Approximate Human ELO     |
# |------------------|---------------------------|
# | 1                | 800                       |
# | 2                | 900                       |
# | 3                | 1000                      |
# | 4                | 1100                      |
# | 5                | 1200                      |
# | 6                | 1300                      |
# | 7                | 1400                      |
# | 8                | 1500                      |
# | 9                | 1600                      |
# | 10               | 1700                      |
# | 11               | 1800                      |
# | 12               | 1900                      |
# | 13               | 2000                      |
# | 14               | 2100                      |
# | 15               | 2200                      |
# | 16               | 2300                      |
# | 17               | 2400                      |
# | 18               | 2500                      |
# | 19               | 2600                      |
# | 20               | 2700+ (Super Grandmaster) |
# -----------------------------------------------

import os
import numpy as np
import tensorflow as tf
import chess
import json
import chess.pgn
from stockfish import Stockfish

# Load the trained model with custom objects
def top_3_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1) if len(y_true.shape) > 1 else y_true
    y_true = tf.cast(y_true, tf.int32)
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    return tf.keras.metrics.top_k_categorical_accuracy(y_true=y_true_one_hot, y_pred=y_pred, k=3)

model_path = "../Transformers/v4/models/checkpoints4/model_checkpoint.h5"
model = tf.keras.models.load_model(model_path, custom_objects={"top_3_accuracy": top_3_accuracy})

# Load the move index map
with open("../Transformers/v4/models/checkpoints4/id_to_move.json", "r") as f:
    move_index_map = json.load(f)
id_to_move = {v: k for k, v in move_index_map.items()}

# Initialize Stockfish
stockfish_path = "../Stockfish/stockfish/stockfish.exe"
stockfish = Stockfish(stockfish_path)
stockfish.set_skill_level(1)

# Initialize PGN game
pgn_game = chess.pgn.Game()
pgn_game.headers["Event"] = "Stockfish vs Transformer"
pgn_game.headers["White"] = "Transformer Model"
pgn_game.headers["Black"] = "Stockfish"

# Set up the node for adding moves
node = pgn_game

# Encode the board state
def encode_board(fen):
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board = chess.Board(fen)
    board_tensor = np.zeros((8, 8, 16), dtype=np.float32)

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

    if board.ep_square:
        row, col = divmod(board.ep_square, 8)
        board_tensor[row, col, 14] = 1.0

    return board_tensor

# Encode the move history (no fixed maximum length)
def encode_history(move_history):
    """
    Encode the move history dynamically, without truncation.
    
    Args:
    - move_history (list): List of UCI moves.
    
    Returns:
    - list: Encoded move history with indices corresponding to the move index map.
    """
    return [move_index_map.get(move, 0) for move in move_history]  # Default to 0 if move not in map

# Validate UCI move
def is_valid_uci_move(move):
    try:
        chess.Move.from_uci(move)
        return True
    except chess.InvalidMoveError:
        return False

def predict_next_move(fen, move_history, policy_weight=0.5, value_weight=0.5):
    """
    Predict the next move using the Transformer model with full move history.

    Args:
    - fen (str): FEN string of the current board position.
    - move_history (list): List of past moves in UCI format.
    - policy_weight (float): Weight assigned to policy output for scoring.
    - value_weight (float): Weight assigned to value output for scoring.

    Returns:
    - str: The best move in UCI format.
    - list: Top legal moves with scores.
    - float: Evaluation value of the best move.
    """
    board = chess.Board(fen)

    # Encode board state and move history
    encoded_board = encode_board(fen)
    encoded_board = np.expand_dims(encoded_board, axis=0)  # Add batch dimension
    encoded_history = encode_history(move_history)
    encoded_history = np.expand_dims(encoded_history, axis=0)  # Add batch dimension

    # Get predictions
    policy_output, value_output = model.predict([encoded_board, encoded_history])
    
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

# Play a game between the model and Stockfish
board = chess.Board()
move_history = []

while not board.is_game_over():
    if board.turn:  # Model's turn (White if True, Black if False)
        fen = board.fen()
        best_move, legal_moves, position_value = predict_next_move(fen, move_history)
        print(f"Transformer Best Move: {best_move}")
        print("Top Legal Moves with Scores:")
        for move, score in legal_moves[:5]:  # Show top 5 moves
            print(f"Move: {move}, Score: {score:.4f}")
        print(f"Position Value: {position_value:.4f}")

        if best_move:
            board.push(chess.Move.from_uci(best_move))
            move_history.append(best_move)
            node = node.add_variation(chess.Move.from_uci(best_move))
        else:
            print(f"No valid move found. Model failed.")
            break
    else:
        stockfish.set_fen_position(board.fen())
        stockfish_move = stockfish.get_best_move()
        print(f"Stockfish's Move: {stockfish_move}")
        stockfish_move_obj = chess.Move.from_uci(stockfish_move)
        if stockfish_move_obj in board.legal_moves:
            board.push(stockfish_move_obj)
            move_history.append(stockfish_move)
            node = node.add_variation(stockfish_move_obj)
        else:
            print("Stockfish produced an invalid move.")
            break

# Output the result
result = board.result()
pgn_game.headers["Result"] = result
print(f"Game Over. Result: {result}")

# Save PGN
with open("game.pgn", "w") as pgn_file:
    pgn_file.write(str(pgn_game))

print("Final PGN:")
print(pgn_game)



"""# Load the trained model
model_path = "../Transformers/v4/models/checkpoints/model_checkpoint.h5"
model = tf.keras.models.load_model(model_path)

# Load the move index map
with open("../Transformers/v4/models/checkpoints/id_to_move.json", "r") as f:
    move_index_map = json.load(f)
id_to_move = {v: k for k, v in move_index_map.items()}

# Initialize Stockfish
stockfish_path = "../Stockfish/stockfish/stockfish.exe"
stockfish = Stockfish(stockfish_path)
stockfish.set_skill_level(1)  # Set Stockfish ELO level (0-20)

# Initialize PGN game
pgn_game = chess.pgn.Game()
pgn_game.headers["Event"] = "Stockfish vs Transformer"
pgn_game.headers["White"] = "Transformer Model"
pgn_game.headers["Black"] = "Stockfish"

# Set up the node for adding moves
node = pgn_game

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
def predict_next_move(fen, policy_weight=0.5, value_weight=0.5):
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

# Initialize the chess board and move history
board = chess.Board()

# Play a game between the model and Stockfish
while not board.is_game_over():
    if board.turn:  # Model's turn (White if True, Black if False)
        fen = board.fen()
        best_move, legal_moves, position_value = predict_next_move(fen)
        print(f"Transformer Best move: {best_move}")
        print("Top legal moves with scores:")
        for move, score in legal_moves[:5]:  # Show top 5 moves
            print(f"Move: {move}, Score: {score:.4f}")
        print(f"Position value: {position_value:.4f}")
        if best_move:
            board.push(chess.Move.from_uci(best_move))
            node = node.add_variation(chess.Move.from_uci(board.peek().uci()))
        else:
            print(f"Invalid move predicted by model: {best_move}")
            break
    else:  # Stockfish's turn
        stockfish.set_fen_position(board.fen())  # Synchronize Stockfish with the current board
        print(f"FEN passed to Stockfish: {board.fen()}")
        stockfish_move = stockfish.get_best_move()
        print(f"Stockfish's move: {stockfish_move}")

        # Convert Stockfish move to chess.Move and validate
        stockfish_move_obj = chess.Move.from_uci(stockfish_move)
        if stockfish_move_obj in board.legal_moves:
            san_move = board.san(stockfish_move_obj)
            board.push(stockfish_move_obj)
            print("San move",san_move)
            node = node.add_variation(stockfish_move_obj)
        else:
            print(f"Invalid move from Stockfish: {stockfish_move}")
            print(f"Legal moves: {[board.san(move) for move in board.legal_moves]}")
            break

# Output the game result
result = board.result()  # Example: "1-0" (White wins), "0-1" (Black wins), "1/2-1/2" (Draw)
print(f"Game over. Result: {result}")

pgn_game.headers["Result"] = result

# Save the PGN to a file
with open("game.pgn", "w") as pgn_file:
    pgn_file.write(str(pgn_game))

# Print the PGN to the console
print("Final PGN:")
print(str(pgn_game))
"""
