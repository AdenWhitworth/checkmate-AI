import chess
import chess.pgn
import sqlite3
import json
from tqdm import tqdm
from stockfish import Stockfish

# Initialize Stockfish
stockfish_path = "../../Stockfish/stockfish/stockfish.exe"
stockfish = Stockfish(stockfish_path, parameters={"Threads": 8, "Skill Level": 10})

# Counters for evaluations
db_eval_count = 0
stockfish_eval_count = 0

def normalize_fen(fen):
    """
    Normalize a FEN string by removing the halfmove clock and fullmove number.
    """
    return " ".join(fen.split()[:4])

def evaluate_missing_fens_with_stockfish(missing_fens):
    """
    Evaluate the missing FENs using Stockfish.
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
    Fetch evaluations for all legal moves from the database.
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
    Save only missing evaluations for a FEN to the database in a single transaction.
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

def fetch_or_evaluate_legal_moves(fen, board, db_conn):
    """
    Fetch or evaluate legal moves for a given FEN and board position.
    """
    global db_eval_count, stockfish_eval_count

    # Fetch all resulting FENs and their evaluations from the database
    resulting_fens, db_evals, missing_fens = fetch_legal_moves_evaluations_from_db(fen, board, db_conn)

    # Increment db_eval_count for all evaluations fetched from the database
    db_eval_count += len(resulting_fens) - len(missing_fens)

    if missing_fens:
        # Evaluate missing FENs using Stockfish
        missing_evals = evaluate_missing_fens_with_stockfish(missing_fens)

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

    return legal_moves, cp_evals, mate_evals

def process_game(game, db_conn):
    """
    Process a single game to extract training data for the midgame phase,
    stopping when the endgame threshold is reached.
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
        legal_moves, cp_evals, mate_evals = fetch_or_evaluate_legal_moves(fen_before_move, board, db_conn)

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

    return midgame_data

def is_endgame(board):
    """
    Determine if the current position is in the endgame phase, 
    defined as having 7 or fewer pieces on the board.
    """
    return len(board.piece_map()) <= 7

def process_pgn_file(pgn_path, db_path, midgame_output):
    """
    Process a PGN file and generate training data for midgame.
    """
    global db_eval_count, stockfish_eval_count  # Access global counters

    print("Begin processing games...")
    db_conn = sqlite3.connect(db_path)
    with open(pgn_path) as pgn, open(midgame_output, "w") as mid_file:
        with tqdm(total=26173, desc="Processing games") as pbar:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                midgame_data = process_game(game, db_conn)
                for entry in midgame_data:
                    mid_file.write(json.dumps(entry) + "\n")
                pbar.update(1)
    db_conn.close()

    # Print evaluation counts
    print("\nProcessing Summary:")
    print(f"Total evaluations from database: {db_eval_count}")
    print(f"Total evaluations from Stockfish: {stockfish_eval_count}")

def process_first_game(pgn_path, db_path, midgame_output, endgame_output):
    """
    Process only the first game in the PGN file for testing.
    """
    db_conn = sqlite3.connect(db_path)
    with open(pgn_path) as pgn, open(midgame_output, "w") as mid_file, open(endgame_output, "w") as end_file:
        game = chess.pgn.read_game(pgn)
        if game:
            midgame_data, endgame_data = process_game(game, db_conn)
            for entry in midgame_data:
                mid_file.write(json.dumps(entry) + "\n")
            for entry in endgame_data:
                end_file.write(json.dumps(entry) + "\n")
    db_conn.close()

# File paths
db_path = r"D:\checkmate_ai\evaluations.db"
pgn_file = "../../PGN Games/partial_lichess_games_26k_filtered_2000_elo.pgn"
midgame_output = r"D:\checkmate_ai\game_phases\midgame_data2.jsonl"

# Run the processing
process_pgn_file(pgn_file, db_path, midgame_output)

# Run the processing for the first game only
#process_first_game(pgn_file, db_path, midgame_output, endgame_output)