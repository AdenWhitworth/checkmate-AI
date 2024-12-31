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

def process_game(game, db_conn):
    global db_eval_count, stockfish_eval_count  # Access global counters

    training_data = []
    board = game.board()
    moves = []

    for move in game.mainline_moves():
        # Capture the FEN before making the move
        fen_before_move = normalize_fen(board.fen())

        # Store the move
        moves.append(move.uci())

        # Make the move on the board
        board.push(move)

        # Evaluate the board position after the move
        fen_after_move = normalize_fen(board.fen())
        eval_data = fetch_evaluation(fen_after_move, db_conn)

        if eval_data is None:
            # Evaluate the FEN after the move using Stockfish with limited depth
            eval_data = evaluate_with_stockfish(fen_after_move)
            if eval_data:
                stockfish_eval_count += 1  # Increment Stockfish evaluation counter
                save_to_database(fen_after_move, eval_data, db_conn)  # Save new evaluation to the database
        else:
            db_eval_count += 1  # Increment database evaluation counter

        if eval_data is not None:
            # Prepare separate CP and Mate values for the training data
            value_cp = eval_data["value"] if eval_data["eval_type"] == "cp" else None
            value_mate = eval_data["value"] if eval_data["eval_type"] == "mate" else None

            # Add the training data entry
            training_data.append({
                "fen": fen_before_move,      # FEN before the move
                "moves": moves[:-1],         # Moves leading up to the current FEN
                "next_move": move.uci(),     # Move played in the game
                "value_cp": value_cp,        # Centipawn evaluation (None if mate)
                "value_mate": value_mate     # Mate evaluation (None if CP)
            })

    return training_data


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
    stockfish.set_depth(10)  # Limit the search depth for faster evaluations
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


def count_games_in_pgn(pgn_path):
    """
    Count the total number of games in the PGN file for progress tracking.
    """
    with open(pgn_path) as pgn:
        return sum(1 for _ in iter(lambda: chess.pgn.read_game(pgn), None))


def process_pgn_file(pgn_path, db_path, output_path):
    """
    Process a PGN file and generate training data with evaluations.
    """
    global db_eval_count, stockfish_eval_count  # Access global counters

    print("Begin processing games...")
    db_conn = sqlite3.connect(db_path)
    with open(pgn_path) as pgn, open(output_path, "w") as output_file:
        with tqdm(total=26173, desc="Processing games") as pbar:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                training_data = process_game(game, db_conn)
                for entry in training_data:
                    output_file.write(json.dumps(entry) + "\n")
                pbar.update(1)
    db_conn.close()

    # Print evaluation counts
    print("\nProcessing Summary:")
    print(f"Total evaluations from database: {db_eval_count}")
    print(f"Total evaluations from Stockfish: {stockfish_eval_count}")


# File paths
db_path = r"D:\checkmate_ai\evaluations.db"
pgn_file = "../../PGN Games/partial_lichess_games_26k_filtered_2000_elo.pgn"
output_file = r"D:\checkmate_ai\evaluated_training_data3.jsonl"

# Run the processing
process_pgn_file(pgn_file, db_path, output_file)
