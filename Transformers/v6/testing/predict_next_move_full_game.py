"""
Chess Move Prediction Script
============================
This script predicts the best chess move based on the current game state (opening, middle game, or endgame).
It leverages trained models for the opening and middle game phases and the Lichess tablebase for the endgame.

Key Features:
-------------
1. **State-Based Predictions**:
   - Determines the game phase (opening, middle game, or endgame) based on the number of moves and remaining pieces.
   - Uses appropriate prediction strategies for each phase.

2. **Model Integration**:
   - Incorporates pre-trained models for opening and middle game predictions.
   - Queries the Lichess tablebase for positions with 7 or fewer pieces.

3. **Flexibility and Extensibility**:
   - Accepts FEN strings and move histories for predictions.
   - Allows dynamic integration of additional models or tablebases.

4. **Outputs**:
   - Returns the best predicted move in UCI format for the given game state.
"""
import sys
import os
import chess
import requests
import chess.pgn
import chess.engine

parent_dir = os.path.abspath("../../") 
sys.path.append(parent_dir)

from v5.testing.predict_next_move import predict_next_move_and_outcome

from v6.testing.predict_next_move import predict_next_move

def query_tablebase(fen):
    """
    Query the Lichess tablebase for positions with 7 or fewer pieces.
    Args:
        fen (str): FEN string of the position.

    Returns:
        dict: Tablebase response or None if query fails.
    """
    url = f"https://tablebase.lichess.ovh/standard?fen={fen}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def predict_end_move(fen):
    """
    Get the best move from the tablebase for a position with 7 or fewer pieces.
    Args:
        fen (str): FEN string of the position.

    Returns:
        str: The best move in UCI format, or None if not available.
    """
    tablebase_data = query_tablebase(fen)
    if tablebase_data and "moves" in tablebase_data:
        best_move = max(tablebase_data["moves"], key=lambda move: move.get("wdl", 0))
        return best_move["uci"]
    return None

def predict_best_move(fen, moves, CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE):
    """
    Predict the best move based on the current game phase.

    This function determines the game state (opening, middle game, or endgame) and selects the best move using:
    - Opening: Pre-trained opening model.
    - Middle Game: Pre-trained middle game model.
    - Endgame: Lichess tablebase for positions with 7 or fewer pieces.

    Args:
        fen (str): The FEN string representing the current board position.
        moves (list): List of moves in UCI format leading up to the current position.
        CHECKPOINT_DIR_OPENING (str): Path to the directory containing the opening phase model and data.
        CHECKPOINT_DIR_MIDDLE (str): Path to the directory containing the middle game phase model and data.

    Returns:
        str: The best move in UCI format, or `None` if no move is found.
    """
    board = chess.Board(fen)
    num_pieces = len(board.piece_map())
    
    # Determine game state
    if len(moves) <= 10:  # Opening phase
        print("Game State: Opening")
        best_move, _ = predict_next_move_and_outcome(fen, moves, CHECKPOINT_DIR_OPENING)
    elif num_pieces <= 7:  # Endgame phase
        print("Game State: Endgame")
        best_move = predict_end_move(fen)
    else:  # Middle game phase
        print("Game State: Middle")
        assessment_type = "weighted"
        best_move = predict_next_move(CHECKPOINT_DIR_MIDDLE, assessment_type, fen, moves)
    
    return best_move

if __name__ == "__main__":
    CHECKPOINT_DIR_OPENING = "../../v5/models/checkpoints"
    CHECKPOINT_DIR_MIDDLE = "../models/checkpoints3"

    fen = "1rbqk1n1/2p5/p7/1P2nppr/2PPp2p/2P1P1PP/1B1Q1P2/R3R2K w KQkq - 0 1"
    move_history = ["d2d4", "e7e5", "g1f3", "e5e4", "c2c4", "f8b4", "b1c3", "b4c3", "b2c3", "d7d5", "d1d2", "h7h6", "f3e5", "f7f6", "e2e3", "g7g5", "a2a4", "f6f5", "g2g3", "h6h5", "f1d3", "a7a6","e1g1", "b7b5", "f1e1", "h5h4", "d3e4", "d5e4", "a4b5", "b8d7",
                "h2h3", "h8h5", "c1b2", "a8b8", "g1h1", "d7e5"]
    best_move = predict_best_move(fen, move_history, CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE)

    print("Best Opening Move: ", best_move)

    fen = "rnbqkb1r/pp2pppp/3p4/8/3nP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 5"
    move_history = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]

    best_move = predict_best_move(fen, move_history, CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE)

    print("Best Middle Move: ", best_move)

    fen = "8/8/8/8/5K2/8/2k5/8 w - - 0 50"
    move_history = [
        "e2e4", "e7e5", "f2f3", "d7d5", "e4d5", "d8d5", "g1f3",
        "d5e4", "f1e2", "e4e2", "e1e2", "b8c6", "f3e5", "c6e5",
        "d2d4", "e5g4", "h2h3", "g4f2", "e2f2", "f7f5", "g2g4",
        "f5g4", "h3g4", "e8e7", "f2e3", "e7d6", "e3e4", "d6c6",
        "f4f5", "h8h7", "e4f4", "g7g6", "f5g6", "h7h8", "g6h7",
        "c6d5", "f4g4", "d5e4", "g4g5", "e4d3", "g5g6", "d3e4",
        "g6g7", "e4d5", "g7h7", "d5e4", "h7h6", "e4d3", "h6h5",
        "d3e2", "h5h4", "e2f1", "h4h3", "f1g1", "h3h2", "g1h1",
        "h2h1q", "h1f3", "h1h1", "f3g2", "h1g1", "g2f3", "g1g2",
        "f3f4", "g2g3", "f4f5", "g3g4", "f5f6", "g4g5", "f6f7",
        "g5g6", "f7f8q", "g6g7", "f8e7", "g7h8", "e7g8", "h8g7",
        "g8h8", "g7h6", "h8g8", "h6g5", "g8f7", "g5f6", "f7f8",
        "f6e5", "f8e7", "e5d6", "e7d8", "d6e7", "d8e8", "e7f8",
        "e8f8"
    ]

    best_move = predict_best_move(fen, move_history, CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE)

    print("Best Ending Move: ", best_move)

