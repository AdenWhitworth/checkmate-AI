"""
Chess Game Simulation: Transformer vs Stockfish
================================================

This script simulates a chess game between the V1 Transformer-based chess model (as White)
and Stockfish (as Black). The game is played using PGN (Portable Game Notation) format,
with moves being recorded for analysis and reproducibility.

Stockfish Skill Levels vs Human ELO (Approximation)
---------------------------------------------------
| Stockfish Level | Approximate Human ELO          |
|------------------|-------------------------------|
| 1                | 800                           |
| 2                | 900                           |
| 3                | 1000                          |
| 4                | 1100                          |
| 5                | 1200                          |
| 6                | 1300                          |
| 7                | 1400                          |
| 8                | 1500                          |
| 9                | 1600                          |
| 10               | 1700                          |
| 11               | 1800                          |
| 12               | 1900                          |
| 13               | 2000                          |
| 14               | 2100                          |
| 15               | 2200                          |
| 16               | 2300                          |
| 17               | 2400                          |
| 18               | 2500                          |
| 19               | 2600                          |
| 20               | 2700+ (Super Grandmaster)     |

Features:
---------
1. **Stockfish Integration**:
   - Initialize Stockfish with skill levels corresponding to human ELO ratings.
   - Predict the best move for Stockfish based on the current board position.

2. **Transformer Model Integration**:
   - Utilize a Transformer-based model to predict the best move during the game.
   - Map predicted moves between UCI (Universal Chess Interface) and SAN (Standard Algebraic Notation) formats.

3. **PGN Support**:
   - Record the entire game in PGN format for post-game analysis and storage.

4. **Error Handling**:
   - Verify the legality of moves before pushing them to the board.
   - Handle invalid moves gracefully and terminate the game if necessary.

Inputs:
-------
- Stockfish Path: Path to the Stockfish executable.
- Transformer Model Path: Path to the ONNX model for the Transformer.
- Move-to-ID Mapping: JSON file mapping moves to unique IDs for the Transformer model.
- ID-to-Move Mapping: JSON file mapping IDs back to moves for the Transformer model.

Outputs:
--------
- PGN File: Saves the completed game in PGN format for future analysis.
- Console Logs: Displays moves and evaluations during the game.

"""

import chess
import chess.pgn
from stockfish import Stockfish
import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
#project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from predict_next_move import setup_and_predict_move

def initialize_stockfish(stockfish_path, skill_level=1):
    """
    Initialize the Stockfish engine.

    Args:
        stockfish_path (str): Path to the Stockfish executable.
        skill_level (int): Skill level of Stockfish (0-20).

    Returns:
        Stockfish: Configured Stockfish engine.
    """
    stockfish = Stockfish(stockfish_path)
    stockfish.set_skill_level(skill_level)
    return stockfish

def initialize_pgn_game():
    """
    Initialize a PGN game with metadata for the event.

    Returns:
        chess.pgn.Game: A new PGN game instance with headers set.
    """
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Stockfish vs Transformer"
    pgn_game.headers["White"] = "Transformer Model"
    pgn_game.headers["Black"] = "Stockfish"
    return pgn_game

def play_transformer_turn(board, move_history, node, model_path, move_to_id_path, id_to_move_path):
    """
    Handle the Transformer's turn to play a move.

    Args:
        board (chess.Board): Current board state.
        move_history (list): List of moves in SAN format.
        node (chess.pgn.GameNode): Current PGN node.
        model_path (str): Path to the Transformer model.
        move_to_id_path (str): Path to the move-to-ID JSON file.
        id_to_move_path (str): Path to the ID-to-move JSON file.

    Returns:
        tuple: Updated PGN node and a boolean indicating if the move was valid.
    """
    predicted_move, _, _, _ = setup_and_predict_move(model_path, move_to_id_path, id_to_move_path, move_history)
    print(f"Model's move: {predicted_move}")
    move_obj = chess.Move.from_uci(predicted_move)

    if move_obj in board.legal_moves:
        san_move = board.san(move_obj)
        board.push(move_obj)
        print(f"Transformer move (UCI: {predicted_move}, SAN: {san_move})")
        move_history.append(san_move)
        node = node.add_variation(move_obj)
        return node, True
    else:
        print(f"Invalid move predicted by Transformer: {predicted_move}")
        return node, False

def play_stockfish_turn(board, move_history, node, stockfish):
    """
    Handle Stockfish's turn to play a move.

    Args:
        board (chess.Board): Current board state.
        move_history (list): List of moves in SAN format.
        node (chess.pgn.GameNode): Current PGN node.
        stockfish (Stockfish): Initialized Stockfish engine.

    Returns:
        tuple: Updated PGN node and a boolean indicating if the move was valid.
    """
    stockfish.set_fen_position(board.fen())
    stockfish_move = stockfish.get_best_move()
    print(f"Stockfish's move: {stockfish_move}")
    move_obj = chess.Move.from_uci(stockfish_move)

    if move_obj in board.legal_moves:
        san_move = board.san(move_obj)
        board.push(move_obj)
        print(f"Stockfish move (UCI: {stockfish_move}, SAN: {san_move})")
        move_history.append(san_move)
        node = node.add_variation(move_obj)
        return node, True
    else:
        print(f"Invalid move from Stockfish: {stockfish_move}")
        return node, False

def save_pgn(pgn_game, file_name="game.pgn"):
    """
    Save the completed PGN game to a file.

    Args:
        pgn_game (chess.pgn.Game): Completed PGN game.
        file_name (str): Name of the file to save the PGN.
    """
    with open(file_name, "w") as pgn_file:
        pgn_file.write(str(pgn_game))
    print(f"Game saved to {file_name}")

def main(stockfish_path, model_path, move_to_id_path, id_to_move_path):
    """
    Main function to run a chess game between Transformer and Stockfish.

    Args:
        stockfish_path (str): Path to the Stockfish executable.
        model_path (str): Path to the Transformer model.
        move_to_id_path (str): Path to the move-to-ID JSON file for the Transformer model.
        id_to_move_path (str): Path to the ID-to-move JSON file for the Transformer model.
    """
    stockfish = initialize_stockfish(stockfish_path, skill_level=1)

    pgn_game = initialize_pgn_game()
    node = pgn_game
    board = chess.Board()
    move_history = []

    while not board.is_game_over():
        if board.turn:  # Transformer's turn (White)
            node, valid_move = play_transformer_turn(board, move_history, node, model_path, move_to_id_path, id_to_move_path)
            if not valid_move:
                break
        else:  # Stockfish's turn (Black)
            node, valid_move = play_stockfish_turn(board, move_history, node, stockfish)
            if not valid_move:
                break

    # Game result
    result = board.result()
    pgn_game.headers["Result"] = result
    print(f"Game over. Result: {result}")

    # Save and display PGN
    save_pgn(pgn_game)
    print("Final PGN:")
    print(pgn_game)

if __name__ == "__main__":
    stockfish_path = "../../../Stockfish/stockfish/stockfish.exe"
    model_path = "../models/base_transformer_50k_games_Models/onnx_models/model_elo_2000_plus.onnx"
    move_to_id_path = "../models/base_transformer_50k_games_Models/move_to_id.json"
    id_to_move_path = "../models/base_transformer_50k_games_Models/id_to_move.json"
    main(stockfish_path, model_path, move_to_id_path, id_to_move_path)



