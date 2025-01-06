"""
Script to check if FEN positions from a chess game exist in an SQLite database.

This script:
1. Normalizes FEN strings to simplify database queries.
2. Checks if each FEN from a game exists in the database.
3. Outputs details about found and missing FENs.

Functions:
- normalize_fen: Normalizes a FEN string by keeping only essential fields.
- check_fens_in_database: Checks if FENs from a single game exist in the database.

Usage:
- Update `db_path` and `pgn_path` to point to your database and PGN file.
- Run the script to analyze the FENs of the first game in the PGN file.

Requirements:
- python-chess library
- SQLite database with a table named `evaluations` containing a `fen` column
"""

import chess
import chess.pgn
import sqlite3


def normalize_fen(fen):
    """
    Normalize a FEN string by keeping only the first four fields.

    Args:
        fen (str): Full FEN string.

    Returns:
        str: Normalized FEN string with only the first four fields.
    """
    return " ".join(fen.split()[:4])

def check_fens_in_database(game, db_path):
    """
    Check if FENs from a single game are present in the database and print results.

    Args:
        game (chess.pgn.Game): A parsed chess game object.
        db_path (str): Path to the SQLite database.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    board = game.board()
    missing_fens = []
    found_fens = []
    total_moves = 0

    for move in game.mainline_moves():
        board.push(move)  # Apply the move to the board
        fen = normalize_fen(board.fen())  # Normalize FEN
        
        # Query the database
        cursor.execute("SELECT COUNT(*) FROM evaluations WHERE fen = ?", (fen,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            missing_fens.append(fen)
        else:
            found_fens.append(fen)
        total_moves += 1

    conn.close()
    
    # Print results
    print(f"Total moves: {total_moves}")
    print(f"Moves found in database: {len(found_fens)}")
    print(f"Moves missing from database: {len(missing_fens)}")
    
    print("\nFound FENs:")
    for fen in found_fens:
        print(fen)
    
    print("\nMissing FENs:")
    for fen in missing_fens:
        print(fen)

if __name__ == "__main__":
    # File paths
    db_path = r"D:\checkmate_ai\evaluations.db"
    pgn_path = "../../PGN Games/pgns/partial_lichess_games_26k_filtered_2000_elo.pgn"

    # Read and process the first game in the PGN file
    with open(pgn_path) as pgn:
        game = chess.pgn.read_game(pgn)
        if game:
            print("Processing the first game from the PGN file...")
            check_fens_in_database(game, db_path)
        else:
            print("No games found in the PGN file.")
