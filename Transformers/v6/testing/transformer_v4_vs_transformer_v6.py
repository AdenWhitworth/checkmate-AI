"""
Interactive Chess Game Simulation

This script simulates a chess game between two Transformer-based AI models:
- **Transformer V5/V6** (White): Uses a model trained on openings and middle-game positions.
- **Transformer 4** (Black): Uses a base Transformer model for move predictions.

Key Features:
- Automatically determines the current game phase (opening, middle-game, endgame).
- Logs the moves of both models and generates a PGN file of the game.
- Allows easy customization of model paths and checkpoints.

Usage:
- Run the script directly, and it will simulate a full game between the two AI models.
- The resulting PGN game is saved to `game.pgn` in the current directory.
"""

import chess
import chess.pgn
import os
import sys

# Dynamically calculate the project root and update `sys.path`
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Import prediction modules
from v4.testing.predict_next_move import predict_next_move
from v6.testing.predict_next_move_full_game import predict_best_move

def initialize_pgn_game():
    """
    Initialize a PGN game with headers for the event and participants.

    Returns:
        chess.pgn.Game: A new PGN game instance with headers set.
    """
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Transformer V5/V6 vs Transformer 4"
    pgn_game.headers["White"] = "Transformer Model V5/V6"
    pgn_game.headers["Black"] = "Transformer Model V4"
    return pgn_game

def play_white_turn(board, move_history, node, checkpoint_opening, checkpoint_middle):
    """
    Handle the White (Transformer V5/V6) turn.

    Args:
        board (chess.Board): The current chess board state.
        move_history (list): List of moves played so far.
        node (chess.pgn.GameNode): The current PGN node.
        checkpoint_opening (str): Path to the opening phase model checkpoints.
        checkpoint_middle (str): Path to the middle game phase model checkpoints.

    Returns:
        tuple: Updated PGN node and a boolean indicating if the move was valid.
    """
    fen = board.fen()
    best_move = predict_best_move(fen, move_history, checkpoint_opening, checkpoint_middle)

    if best_move:
        print(f"Transformer Best Move (White): {best_move}")
        move_obj = chess.Move.from_uci(best_move)
        board.push(move_obj)
        move_history.append(best_move)
        node = node.add_variation(move_obj)
        return node, True
    else:
        print("No valid move found for the Transformer model.")
        return node, False

def play_black_turn(board, move_history, node, model_path, id_to_move_path):
    """
    Handle the Black (Transformer V4) turn.

    Args:
        board (chess.Board): The current chess board state.
        move_history (list): List of moves played so far.
        node (chess.pgn.GameNode): The current PGN node.
        model_path (str): Path to the V4 Transformer h5 model.
        id_to_move_path (str): Path to the ID-to-move JSON mapping.

    Returns:
        tuple: Updated PGN node and a boolean indicating if the move was valid.
    """
    best_move, _, _ = predict_next_move(board.fen(), move_history, model_path, id_to_move_path)
    print(f"Transformer V4 Predicted Move (Black): {best_move}")

    try:
        move_obj = chess.Move.from_uci(best_move)
        if move_obj in board.legal_moves:
            print(f"V4 move {best_move} is legal.")
            board.push(move_obj)
            move_history.append(best_move)
            node = node.add_variation(move_obj)
            return node, True
        else:
            print(f"V4 move {best_move} is illegal.")
            return node, False
    except ValueError:
        print(f"V4 produced an invalid move format: {best_move}")
        return node, False

def save_pgn(pgn_game, file_name="game.pgn"):
    """
    Save the completed PGN game to a file.

    Args:
        pgn_game (chess.pgn.Game): The completed PGN game.
        file_name (str): The file name for saving the PGN.
    """
    with open(file_name, "w") as pgn_file:
        pgn_file.write(str(pgn_game))
    print(f"Game saved to {file_name}")

def main(CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE, MODEL_PATH, ID_TO_MOVE_PATH):
    """
    Main function to simulate a chess game between Transformer V5/V6 (White) and Transformer V4 (Black).

    Args:
        CHECKPOINT_DIR_OPENING (str): Path to the opening phase model checkpoints.
        CHECKPOINT_DIR_MIDDLE (str): Path to the middle game phase model checkpoints.
        MODEL_PATH (str): Path to the V4 Transformer h5 model.
        ID_TO_MOVE_PATH (str): Path to the ID-to-move JSON mapping.
    """
    board = chess.Board()
    move_history = []
    pgn_game = initialize_pgn_game()
    node = pgn_game

    while not board.is_game_over():
        if board.turn:  # White's turn
            node, valid_move = play_white_turn(board, move_history, node, CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE)
            if not valid_move:
                break
        else:  # Black's turn
            node, valid_move = play_black_turn(board, move_history, node, MODEL_PATH, ID_TO_MOVE_PATH)
            if not valid_move:
                break

    # Output the result
    result = board.result()
    pgn_game.headers["Result"] = result
    print(f"Game Over. Result: {result}")
    save_pgn(pgn_game)

if __name__ == "__main__":
    CHECKPOINT_DIR_OPENING = "../../v5/models/checkpoints"
    CHECKPOINT_DIR_MIDDLE = "../models/checkpoints3"
    MODEL_PATH = "../../v4/models/checkpoints4/model_checkpoint.h5"
    ID_TO_MOVE_PATH = "../../v4/models/checkpoints4/id_to_move.json"

    main(CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE, MODEL_PATH, ID_TO_MOVE_PATH)
