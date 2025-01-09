"""
Chess PGN Processing and Evaluation with Stockfish

This script processes chess games stored in PGN files to generate training data for midgame and endgame analysis. 
The script uses Stockfish to evaluate positions and stores the results in a SQLite database for future reuse. 
Midgame and endgame data are written to separate output files.

Main Features:
1. **Stockfish Integration**: Evaluates positions using the Stockfish engine for centipawn and mate scores.
2. **SQLite Database**: Caches evaluations to minimize redundant computations.
3. **Midgame/Endgame Separation**: Differentiates between midgame and endgame phases based on piece count, pawn structure, and king activity.
4. **PGN Parsing**: Processes games move by move, extracting FENs, evaluations, and game metadata.
5. **Data Export**: Outputs midgame and endgame data in JSONL format for further analysis or training machine learning models.

Key Functions:
- `initialize_stockfish`: Initializes the Stockfish engine with configurable parameters.
- `normalize_fen`: Normalizes FEN strings for database storage and comparison.
- `fetch_evaluation`: Retrieves evaluations from the SQLite database.
- `evaluate_with_stockfish`: Evaluates positions using Stockfish with a specified depth.
- `save_to_database`: Saves evaluation results to the SQLite database.
- `is_endgame`: Determines whether a position qualifies as an endgame.
- `process_game`: Processes a single game to extract midgame and endgame data entries.
- `count_games_in_pgn`: Counts the total number of games in a PGN file.
- `process_pgn_file`: Orchestrates the processing of a PGN file, including database queries, Stockfish evaluations, and data export.

Usage:
1. Ensure the following dependencies are installed:
   - `python-chess`
   - `tqdm`
   - `stockfish`
   - `sqlite3`
2. Update the paths to the PGN file, SQLite database, and Stockfish executable.
3. Run the script to process the PGN file and generate midgame and endgame data.
"""
import chess
import chess.pgn
import sqlite3
import json
from tqdm import tqdm
from stockfish import Stockfish

def initialize_stockfish(stockfish_path, threads=8, skill_level=10):
    """
    Initialize the Stockfish engine with given parameters.

    Args:
        stockfish_path (str): Path to the Stockfish executable.
        threads (int): Number of threads to use.
        skill_level (int): Skill level of Stockfish (1-20).

    Returns:
        Stockfish: Initialized Stockfish engine.
    """
    return Stockfish(stockfish_path, parameters={"Threads": threads, "Skill Level": skill_level})

def normalize_fen(fen):
    """
    Normalize a FEN string by removing the halfmove clock and fullmove number.

    Args:
        fen (str): FEN string to normalize.

    Returns:
        str: Normalized FEN string.
    """
    return " ".join(fen.split()[:4])

def fetch_evaluation(fen, db_conn):
    """
    Fetch evaluation data for a FEN from the SQLite database.

    Args:
        fen (str): FEN string to query.
        db_conn (sqlite3.Connection): SQLite database connection.

    Returns:
        dict or None: Evaluation data if found, otherwise None.
    """
    cursor = db_conn.cursor()
    cursor.execute("SELECT eval_type, value FROM evaluations WHERE fen = ?", (fen,))
    result = cursor.fetchone()
    return {"eval_type": result[0], "value": result[1]} if result else None

def evaluate_with_stockfish(fen, stockfish, depth=7):
    """
    Evaluate a FEN using Stockfish with limited depth.

    Args:
        fen (str): FEN string to evaluate.
        stockfish (Stockfish): Stockfish engine instance.
        depth (int): Depth for Stockfish evaluation.

    Returns:
        dict or None: Evaluation data with type and value, or None if invalid.
    """
    stockfish.set_fen_position(fen)
    stockfish.set_depth(depth)
    evaluation = stockfish.get_evaluation()
    if "mate" in evaluation:
        return {"eval_type": "mate", "value": evaluation["mate"]}
    elif "value" in evaluation:
        return {"eval_type": "cp", "value": evaluation["value"]}
    return None

def save_to_database(fen, eval_data, db_conn):
    """
    Save Stockfish evaluation to the SQLite database.

    Args:
        fen (str): FEN string of the evaluated position.
        eval_data (dict): Evaluation data to save.
        db_conn (sqlite3.Connection): SQLite database connection.
    """
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO evaluations (fen, eval_type, value) VALUES (?, ?, ?)",
        (fen, eval_data["eval_type"], eval_data["value"])
    )
    db_conn.commit()

def is_endgame(board):
    """
    Determine if the current position is in the endgame phase.

    Args:
        board (chess.Board): Board position to evaluate.

    Returns:
        bool: True if the position is in the endgame phase, otherwise False.
    """
    pieces = board.piece_map()
    pawns = sum(1 for piece in pieces.values() if piece.piece_type == chess.PAWN)
    major_pieces = sum(1 for piece in pieces.values() if piece.piece_type in [chess.QUEEN, chess.ROOK])

    # Primary criteria: few pieces left
    total_pieces = len(pieces)
    if total_pieces <= 8 and major_pieces <= 1:
        return True

    # Secondary criteria: pawn structures
    if pawns >= (total_pieces / 2) and major_pieces == 0:
        return True

    # Secondary criteria: king activity
    king_positions = [board.king(color) for color in [chess.WHITE, chess.BLACK]]
    for king_pos in king_positions:
        if chess.square_distance(king_pos, chess.E4) <= 2:  # King is near the center
            return True

    return False

def process_game(game, db_conn, stockfish, db_eval_count=0, stockfish_eval_count=0):
    """
    Process a single game to extract training data for midgame and endgame.

    Args:
        game (chess.pgn.Game): PGN game object to process.
        db_conn (sqlite3.Connection): SQLite database connection.
        stockfish (Stockfish): Stockfish engine instance.
        db_eval_count (int): Current database evaluation count.
        stockfish_eval_count (int): Current Stockfish evaluation count.

    Returns:
        tuple: Lists of midgame and endgame data entries, updated counts for database and Stockfish evaluations.
    """

    midgame_data = []
    endgame_data = []
    board = game.board()
    moves = []  # Track all moves made in the game

    for move_number, move in enumerate(game.mainline_moves(), start=1):
        # Skip opening moves
        if move_number <= 10:
            board.push(move)
            moves.append(move.uci())
            continue

        # Record FEN before the move
        fen_before_move = normalize_fen(board.fen())

        # Store the move in the game
        moves.append(move.uci())

        # Make the move on the board
        board.push(move)

        # Record FEN after the move
        fen_after_move = normalize_fen(board.fen())

        # Fetch evaluation for the position after the move
        eval_data = fetch_evaluation(fen_after_move, db_conn)
        if eval_data is None:
            # Evaluate the position using Stockfish
            eval_data = evaluate_with_stockfish(fen_after_move, stockfish)
            if eval_data:
                stockfish_eval_count += 1  # Increment Stockfish evaluation counter
                save_to_database(fen_after_move, eval_data, db_conn)  # Save to database
        else:
            db_eval_count += 1  # Increment database evaluation counter

        # Include evaluation data only if available
        if eval_data is not None:
            # Separate raw centipawn and mate values
            value_cp = eval_data["value"] if eval_data["eval_type"] == "cp" else None
            value_mate = eval_data["value"] if eval_data["eval_type"] == "mate" else None

            entry = {
                "fen": fen_before_move,  # FEN before the move
                "moves": moves[:-1],     # All moves leading up to this position
                "next_move": move.uci(), # Grandmaster's move
                "value_cp": value_cp,    # Centipawn evaluation (raw value, or None)
                "value_mate": value_mate # Mate evaluation (raw value, or None)
            }

            # Separate midgame and endgame data
            if is_endgame(board):
                endgame_data.append(entry)
            else:
                midgame_data.append(entry)

    return midgame_data, endgame_data, db_eval_count, stockfish_eval_count

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

def process_pgn_file(pgn_path, db_path, midgame_output, endgame_output, stockfish_path):
    """
    Process a PGN file and generate training data for midgame and endgame.

    Args:
        pgn_path (str): Path to the PGN file.
        db_path (str): Path to the SQLite database file.
        midgame_output (str): Path to the midgame data output file.
        endgame_output (str): Path to the endgame data output file.
        stockfish_path (str): Path to the Stockfish engine executable.

    Returns:
        None
    """
    db_eval_count = 0
    stockfish_eval_count = 0

    stockfish = initialize_stockfish(stockfish_path)
    
    print("Counting total PGN Games")
    total_games = count_games_in_pgn(pgn_path)

    print("Begin processing games...")
    db_conn = sqlite3.connect(db_path)
    with open(pgn_path) as pgn, open(midgame_output, "w") as mid_file, open(endgame_output, "w") as end_file:
        with tqdm(total=total_games, desc="Processing games") as pbar:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                midgame_data, endgame_data, db_eval_count, stockfish_eval_count = process_game(game, db_conn, stockfish, db_eval_count, stockfish_eval_count)
                for entry in midgame_data:
                    mid_file.write(json.dumps(entry) + "\n")
                for entry in endgame_data:
                    end_file.write(json.dumps(entry) + "\n")
                pbar.update(1)
    db_conn.close()

    # Print evaluation counts
    print("\nProcessing Summary:")
    print(f"Total evaluations from database: {db_eval_count}")
    print(f"Total evaluations from Stockfish: {stockfish_eval_count}")

if __name__ == "__main__":
    DB_PATH = r"D:\checkmate_ai\evaluations.db"
    PGN_FILE = "../../../PGN Games/pgnspartial_lichess_games_26k_filtered_2000_elo.pgn"
    MIDGAME_OUTPUT = r"D:\checkmate_ai\game_phases\midgame_data.jsonl"
    ENDGAME_OUTPUT = r"D:\checkmate_ai\game_phases\endgame_data.jsonl"
    STOCKFISH_PATH = "../../../Stockfish/stockfish/stockfish.exe"

    process_pgn_file(PGN_FILE, DB_PATH, MIDGAME_OUTPUT, ENDGAME_OUTPUT, STOCKFISH_PATH)
