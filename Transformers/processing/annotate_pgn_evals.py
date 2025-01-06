"""
Script to process chess games in PGN format, evaluate positions using Stockfish, 
and save the results as training data for machine learning.

This script:
1. Reads chess games from a PGN file.
2. Normalizes FEN strings and evaluates board positions using an SQLite database or Stockfish.
3. Saves evaluations to the database and outputs training data in JSONL format.

Functions:
- process_game: Processes a single chess game to generate training data.
- normalize_fen: Normalizes a FEN string by removing move counters.
- fetch_evaluation: Fetches evaluation data from the SQLite database.
- evaluate_with_stockfish: Evaluates a FEN using Stockfish with limited depth.
- save_to_database: Saves evaluation data to the SQLite database.
- count_games_in_pgn: Counts the total number of games in a PGN file for progress tracking.
- process_pgn_file: Main function to process a PGN file and generate training data.

Requirements:
- python-chess
- tqdm
- stockfish (Python wrapper)

Usage:
- Update the file paths for `db_path`, `pgn_file`, and `output_file`.
- Ensure the SQLite database and Stockfish executable are properly configured.
"""

import chess
import chess.pgn
import sqlite3
import json
from tqdm import tqdm
from stockfish import Stockfish

# Counters for evaluations
db_eval_count = 0
stockfish_eval_count = 0

def normalize_fen(fen):
    """
    Normalize a FEN string by removing the halfmove clock and fullmove number.

    Args:
        fen (str): Full FEN string.

    Returns:
        str: Normalized FEN string (first four fields only).
    """
    return " ".join(fen.split()[:4])


def fetch_evaluation(fen, db_conn):
    """
    Fetch evaluation data for a FEN from the SQLite database.

    Args:
        fen (str): Normalized FEN string.
        db_conn (sqlite3.Connection): SQLite connection object.

    Returns:
        dict or None: Evaluation data if found, otherwise None.
    """
    cursor = db_conn.cursor()
    cursor.execute("SELECT eval_type, value FROM evaluations WHERE fen = ?", (fen,))
    result = cursor.fetchone()
    return {"eval_type": result[0], "value": result[1]} if result else None


def evaluate_with_stockfish(stockfish, fen):
    """
    Evaluate a FEN using Stockfish with limited depth.

    Args:
        fen (str): Normalized FEN string.

    Returns:
        dict or None: Evaluation data as a dictionary, or None if invalid.
    """
    stockfish.set_fen_position(fen)
    stockfish.set_depth(10)  # Limit the search depth for faster evaluations
    evaluation = stockfish.get_evaluation()

    if "mate" in evaluation:
        return {"eval_type": "mate", "value": evaluation["mate"]}
    elif "value" in evaluation:
        return {"eval_type": "cp", "value": evaluation["value"]}
    return None


def save_to_database(fen, eval_data, db_conn):
    """
    Save Stockfish evaluation data to the SQLite database.

    Args:
        fen (str): Normalized FEN string.
        eval_data (dict): Evaluation data to save.
        db_conn (sqlite3.Connection): SQLite connection object.
    """
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO evaluations (fen, eval_type, value) VALUES (?, ?, ?)",
        (fen, eval_data["eval_type"], eval_data["value"]),
    )
    db_conn.commit()

def process_game(game, db_conn, stockfish):
    """
    Process a single chess game to generate training data.

    Args:
        game (chess.pgn.Game): Parsed chess game object.
        db_conn (sqlite3.Connection): SQLite connection object.

    Returns:
        list: List of training data dictionaries.
    """
    global db_eval_count, stockfish_eval_count

    training_data = []
    board = game.board()
    moves = []

    for move in game.mainline_moves():
        fen_before_move = normalize_fen(board.fen())
        moves.append(move.uci())
        board.push(move)
        fen_after_move = normalize_fen(board.fen())

        eval_data = fetch_evaluation(fen_after_move, db_conn)
        if eval_data is None:
            eval_data = evaluate_with_stockfish(stockfish, fen_after_move)
            if eval_data:
                stockfish_eval_count += 1
                save_to_database(fen_after_move, eval_data, db_conn)
        else:
            db_eval_count += 1

        if eval_data is not None:
            value_cp = eval_data["value"] if eval_data["eval_type"] == "cp" else None
            value_mate = eval_data["value"] if eval_data["eval_type"] == "mate" else None
            training_data.append({
                "fen": fen_before_move,
                "moves": moves[:-1],
                "next_move": move.uci(),
                "value_cp": value_cp,
                "value_mate": value_mate,
            })

    return training_data

def count_games_in_pgn(pgn_path):
    """
    Count the total number of games in a PGN file.

    Args:
        pgn_path (str): Path to the PGN file.

    Returns:
        int: Total number of games in the file.
    """
    with open(pgn_path) as pgn:
        return sum(1 for _ in iter(lambda: chess.pgn.read_game(pgn), None))

def process_pgn_file(pgn_path, db_path, output_path, stockfish_path):
    """
    Process a PGN file to generate training data with evaluations.

    Args:
        pgn_path (str): Path to the PGN file.
        db_path (str): Path to the SQLite database.
        output_path (str): Path to save the training data in JSONL format.
    """
    global db_eval_count, stockfish_eval_count

    print("Begin processing games...")
    db_conn = sqlite3.connect(db_path)
    stockfish = Stockfish(stockfish_path, parameters={"Threads": 8, "Skill Level": 10})
    with open(pgn_path) as pgn, open(output_path, "w") as output_file:
        with tqdm(total=count_games_in_pgn(pgn_path), desc="Processing games") as pbar:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                training_data = process_game(game, db_conn, stockfish)
                for entry in training_data:
                    output_file.write(json.dumps(entry) + "\n")
                pbar.update(1)
    db_conn.close()

    print("\nProcessing Summary:")
    print(f"Total evaluations from database: {db_eval_count}")
    print(f"Total evaluations from Stockfish: {stockfish_eval_count}")

# Run the processing
if __name__ == "__main__":

    # Initialize paths
    stockfish_path = "../../Stockfish/stockfish/stockfish.exe"
    db_path = r"D:\checkmate_ai\evaluations.db"
    pgn_file = "../../PGN Games/partial_lichess_games_26k_filtered_2000_elo.pgn"
    output_file = r"D:\checkmate_ai\evaluated_training_data3.jsonl"

    process_pgn_file(pgn_file, db_path, output_file, stockfish_path)
