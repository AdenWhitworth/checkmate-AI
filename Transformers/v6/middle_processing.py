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

def fetch_evaluation(fen, db_conn):
    """
    Fetch evaluation data for a FEN from the SQLite database.
    """
    cursor = db_conn.cursor()
    cursor.execute("SELECT eval_type, value FROM evaluations WHERE fen = ?", (fen,))
    result = cursor.fetchone()
    return {"eval_type": result[0], "value": result[1]} if result else None

def evaluate_with_stockfish(fen):
    """
    Evaluate a FEN using Stockfish with limited depth.
    """
    stockfish.set_fen_position(fen)
    stockfish.set_depth(7)  # Limit the search depth for faster evaluations
    evaluation = stockfish.get_evaluation()

    if "mate" in evaluation:
        return {"eval_type": "mate", "value": evaluation["mate"]}
    elif "value" in evaluation:
        return {"eval_type": "cp", "value": evaluation["value"]}
    return None

def save_to_database(fen, eval_data, db_conn):
    """
    Save Stockfish evaluation to the SQLite database.
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
    """
    pieces = board.piece_map()
    pawns = sum(1 for piece in pieces.values() if piece.piece_type == chess.PAWN)
    major_pieces = sum(1 for piece in pieces.values() if piece.piece_type in [chess.QUEEN, chess.ROOK])
    minor_pieces = sum(1 for piece in pieces.values() if piece.piece_type in [chess.BISHOP, chess.KNIGHT])

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

def process_game(game, db_conn):
    """
    Process a single game to extract training data for both midgame and endgame.
    """
    global db_eval_count, stockfish_eval_count  # Access global counters

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
            eval_data = evaluate_with_stockfish(fen_after_move)
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

    return midgame_data, endgame_data

def process_pgn_file(pgn_path, db_path, midgame_output, endgame_output):
    """
    Process a PGN file and generate training data for midgame and endgame.
    """
    global db_eval_count, stockfish_eval_count  # Access global counters

    print("Begin processing games...")
    db_conn = sqlite3.connect(db_path)
    with open(pgn_path) as pgn, open(midgame_output, "w") as mid_file, open(endgame_output, "w") as end_file:
        with tqdm(total=26173, desc="Processing games") as pbar:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                midgame_data, endgame_data = process_game(game, db_conn)
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

# File paths
db_path = r"D:\checkmate_ai\evaluations.db"
pgn_file = "../../PGN Games/partial_lichess_games_26k_filtered_2000_elo.pgn"
midgame_output = r"D:\checkmate_ai\game_phases\midgame_data.jsonl"
endgame_output = r"D:\checkmate_ai\game_phases\endgame_data.jsonl"

# Run the processing
process_pgn_file(pgn_file, db_path, midgame_output, endgame_output)
