"""
Chess Data Processing and Evaluation with Stockfish

This script processes chess PGN (Portable Game Notation) files to generate training data for 
machine learning models. It evaluates chess positions using Stockfish and stores evaluation data 
in a SQLite database. The script identifies midgame positions, evaluates legal moves, and generates 
structured data for further analysis or training.

Features:
1. **Stockfish Integration**:
   - Evaluates chess positions using Stockfish.
   - Fetches evaluations from a SQLite database when available, reducing redundant computations.
   - Saves missing evaluations to the database.

2. **PGN Processing**:
   - Processes games from PGN files to extract midgame positions.
   - Identifies legal moves and their evaluations (centipawn or mate).
   - Determines whether a position qualifies as an endgame.

3. **Database Integration**:
   - Stores and retrieves evaluations for chess positions (FEN strings).
   - Ensures efficient re-use of previously computed evaluations.

4. **Training Data Generation**:
   - Outputs midgame data in JSONL format, including FEN, legal moves, evaluations, and next move.

Modules and Functions:
- `initialize_stockfish`: Initializes the Stockfish chess engine with specified parameters.
- `normalize_fen`: Normalizes a FEN string by removing unnecessary components.
- `evaluate_missing_fens_with_stockfish`: Evaluates positions that are not available in the database using Stockfish.
- `fetch_legal_moves_evaluations_from_db`: Fetches evaluations for all legal moves from the database.
- `save_missing_evaluations_to_db`: Saves newly computed evaluations to the database.
- `fetch_or_evaluate_legal_moves`: Combines database fetching and Stockfish evaluation for legal moves.
- `process_game`: Processes a single chess game to extract midgame training data.
- `is_endgame`: Determines if a position is in the endgame phase based on piece count.
- `count_games_in_pgn`: Counts the number of games in a PGN file.
- `process_pgn_file`: Main function to process a PGN file and generate training data.

Usage:
- Configure the paths to the database, PGN file, output file, and Stockfish executable.
- Run the script to process the PGN file, evaluate positions, and generate output.
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

def evaluate_missing_fens_with_stockfish(missing_fens, stockfish):
    """
    Evaluate a list of missing FENs using Stockfish.

    Args:
        missing_fens (list): List of FEN strings to evaluate.
        stockfish (Stockfish): Initialized Stockfish engine instance.

    Returns:
        dict: Mapping of FENs to their evaluation results. 
              Each evaluation result is a dictionary with keys "cp" (centipawn value) 
              and "mate" (mate value or None if not applicable).
    """
    missing_evals = {}
    for fen in missing_fens:
        stockfish.set_fen_position(fen)
        try:
            stockfish.set_depth(7)
            evaluation = stockfish.get_evaluation()
            evals = {"cp": None, "mate": None}
            if "mate" in evaluation:
                evals["mate"] = evaluation["mate"]
            elif "value" in evaluation:
                evals["cp"] = evaluation["value"]
            missing_evals[fen] = evals
        except Exception as e:
            print(f"Error evaluating FEN: {fen}. Error: {e}")
            missing_evals[fen] = {"cp": None, "mate": None}
    return missing_evals

def fetch_legal_moves_evaluations_from_db(fen, board, db_conn):
    """
    Fetch evaluations for all legal moves resulting from a given FEN.

    Args:
        fen (str): Current FEN string.
        board (chess.Board): Chess board instance at the given FEN.
        db_conn (sqlite3.Connection): SQLite database connection.

    Returns:
        tuple: 
            - resulting_fens (dict): Mapping of legal moves (UCI) to resulting FENs.
            - db_evals (dict): Evaluation data for each resulting FEN from the database.
            - missing_fens (list): List of FENs that were not found in the database.
    """
    legal_moves = [m.uci() for m in board.legal_moves]
    
    resulting_fens = {}
    for move in legal_moves:
        board.push(chess.Move.from_uci(move))
        resulting_fens[move] = normalize_fen(board.fen())
        board.pop()

    # Query the database for these resulting FENs
    placeholders = ",".join("?" for _ in resulting_fens.values())
    query = f"SELECT fen, eval_type, value FROM evaluations WHERE fen IN ({placeholders})"
    cursor = db_conn.cursor()
    cursor.execute(query, tuple(resulting_fens.values()))

    # Organize evaluations by FEN
    db_evals = {fen: {"cp": None, "mate": None} for fen in resulting_fens.values()}
    for fen, eval_type, value in cursor.fetchall():
        if eval_type == "cp":
            db_evals[fen]["cp"] = value
        elif eval_type == "mate":
            db_evals[fen]["mate"] = value

    # Determine which FENs still need evaluation
    missing_fens = [
        fen for fen, evals in db_evals.items() if evals["cp"] is None and evals["mate"] is None
    ]

    return resulting_fens, db_evals, missing_fens

def save_missing_evaluations_to_db(missing_evals, db_conn):
    """
    Save evaluations for missing FENs to the SQLite database.

    Args:
        missing_evals (dict): Mapping of FENs to their evaluation results.
        db_conn (sqlite3.Connection): SQLite database connection.

    Returns:
        None
    """

    cursor = db_conn.cursor()
    insert_data = []
    for fen, evals in missing_evals.items():
        if evals["cp"] is not None:
            insert_data.append((fen, "cp", evals["cp"]))
        if evals["mate"] is not None:
            insert_data.append((fen, "mate", evals["mate"]))

    cursor.executemany(
        "INSERT OR IGNORE INTO evaluations (fen, eval_type, value) VALUES (?, ?, ?)",
        insert_data,
    )
    db_conn.commit()

def fetch_or_evaluate_legal_moves(fen, board, db_conn, stockfish, db_eval_count, stockfish_eval_count):
    """
    Fetch or evaluate the evaluations of all legal moves for a given position.

    Args:
        fen (str): Current FEN string.
        board (chess.Board): Chess board instance at the given FEN.
        db_conn (sqlite3.Connection): SQLite database connection.
        stockfish (Stockfish): Initialized Stockfish engine instance.
        db_eval_count (int): Counter for evaluations fetched from the database.
        stockfish_eval_count (int): Counter for evaluations performed by Stockfish.

    Returns:
        tuple:
            - legal_moves (list): List of legal moves (UCI).
            - cp_evals (list): List of centipawn evaluations for each legal move.
            - mate_evals (list): List of mate evaluations for each legal move.
            - db_eval_count (int): Updated count of database evaluations.
            - stockfish_eval_count (int): Updated count of Stockfish evaluations.
    """
    # Fetch all resulting FENs and their evaluations from the database
    resulting_fens, db_evals, missing_fens = fetch_legal_moves_evaluations_from_db(fen, board, db_conn)

    # Increment db_eval_count for all evaluations fetched from the database
    db_eval_count += len(resulting_fens) - len(missing_fens)

    if missing_fens:
        # Evaluate missing FENs using Stockfish
        missing_evals = evaluate_missing_fens_with_stockfish(missing_fens, stockfish)

        # Save missing evaluations to the database
        save_missing_evaluations_to_db(missing_evals, db_conn)

        # Increment stockfish_eval_count for the number of missing evaluations
        stockfish_eval_count += len(missing_evals)

        # Update db_evals with the new evaluations
        for fen, evals in missing_evals.items():
            db_evals[fen] = evals

    # Map evaluations back to their corresponding moves
    legal_moves = list(resulting_fens.keys())
    cp_evals = [db_evals[resulting_fens[move]]["cp"] or 0 for move in legal_moves]
    mate_evals = [db_evals[resulting_fens[move]]["mate"] or 0 for move in legal_moves]

    return legal_moves, cp_evals, mate_evals, db_eval_count, stockfish_eval_count

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
        tuple: Lists of midgame data entries, updated counts for database and Stockfish evaluations.
    """
    midgame_data = []
    board = game.board()
    moves = []

    for move_number, move in enumerate(game.mainline_moves(), start=1):
        # Skip opening moves
        if move_number <= 10:
            board.push(move)
            continue

        # Record FEN before the move
        fen_before_move = normalize_fen(board.fen())

        # Check if it's the endgame phase
        if is_endgame(board):
            break

        # Fetch or evaluate legal moves
        legal_moves, cp_evals, mate_evals, db_eval_count, stockfish_eval_count = fetch_or_evaluate_legal_moves(fen_before_move, board, db_conn, stockfish, db_eval_count, stockfish_eval_count)

        moves.append(move.uci())

        # Make the move
        board.push(move)

        # Create data entry
        entry = {
            "fen": fen_before_move,
            "moves": moves[:-1],
            "legal_moves": legal_moves,
            "cp_evals": cp_evals,
            "mate_evals": mate_evals,
            "next_move": move.uci(),
        }

        midgame_data.append(entry)

    return midgame_data, db_eval_count, stockfish_eval_count

def is_endgame(board):
    """
    Determine if the current position is in the endgame phase, 
    defined as having 7 or fewer pieces on the board.
    """
    return len(board.piece_map()) <= 7

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

def process_pgn_file(pgn_path, db_path, midgame_output, stockfish_path):
    """
    Process a PGN file and generate training data for midgame and endgame.

    Args:
        pgn_path (str): Path to the PGN file.
        db_path (str): Path to the SQLite database file.
        midgame_output (str): Path to the midgame data output file.
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
    with open(pgn_path) as pgn, open(midgame_output, "w") as mid_file:
        with tqdm(total=total_games, desc="Processing games") as pbar:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                midgame_data, db_eval_count, stockfish_eval_count = process_game(game, db_conn, stockfish, db_eval_count, stockfish_eval_count)
                for entry in midgame_data:
                    mid_file.write(json.dumps(entry) + "\n")
                pbar.update(1)
    db_conn.close()

    # Print evaluation counts
    print("\nProcessing Summary:")
    print(f"Total evaluations from database: {db_eval_count}")
    print(f"Total evaluations from Stockfish: {stockfish_eval_count}")

if __name__ == "__main__":
    DB_PATH = r"D:\checkmate_ai\evaluations.db"
    PGN_FILE = "../../../PGN Games/pgnspartial_lichess_games_26k_filtered_2000_elo.pgn"
    MIDGAME_OUTPUT = r"D:\checkmate_ai\game_phases\midgame_data3.jsonl"
    STOCKFISH_PATH = "../../../Stockfish/stockfish/stockfish.exe"

    process_pgn_file(PGN_FILE, DB_PATH, MIDGAME_OUTPUT, STOCKFISH_PATH)


