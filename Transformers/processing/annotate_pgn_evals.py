"""Processing Summary:
Total evaluations from database: 395990
Total evaluations from Stockfish: 1579957"""

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

def evaluate_with_top_moves(stockfish, board, num_moves=5):
    """
    Use Stockfish to get evaluations for the top N legal moves.
    """
    stockfish.set_fen_position(board.fen())
    top_moves = stockfish.get_top_moves(num_moves)

    legal_moves_data = []
    for move in top_moves:
        eval_type = "cp" if "Centipawn" in move else "mate"
        value = move.get("Centipawn", move.get("Mate"))  # Get Centipawn or Mate value

        if value is not None:  # Only include moves with valid evaluations
            legal_moves_data.append({
                "move": move["Move"],
                "eval_type": eval_type,
                "value": value / 100 if eval_type == "cp" else value
            })
        else:
            print(f"Skipping move {move['Move']} due to missing evaluation.")

    return legal_moves_data

def process_game(game, db_conn):
    """
    Process a single PGN game and generate training data, including evaluations for all legal moves.
    """
    global db_eval_count, stockfish_eval_count  # Access global counters

    training_data = []
    board = game.board()
    moves = []

    for move in game.mainline_moves():
        # Capture the FEN before making the move
        fen_before_move = normalize_fen(board.fen())

        # Use Stockfish to evaluate top legal moves
        legal_moves_data = evaluate_with_top_moves(stockfish, board)

        # Store the move played in the game
        moves.append(move.uci())

        # Ensure the grandmaster's move is included in legal_moves_data
        if not any(m["move"] == move.uci() for m in legal_moves_data):
            # If for some reason it's missing, explicitly add it
            board.push(move)  # Temporarily apply the move to get the FEN
            grandmaster_fen = normalize_fen(board.fen())
            board.pop()  # Undo the move

            grandmaster_eval = fetch_evaluation(grandmaster_fen, db_conn)
            if grandmaster_eval is None:
                grandmaster_eval = evaluate_with_stockfish(grandmaster_fen)
                if grandmaster_eval:
                    stockfish_eval_count += 1
                    save_to_database(grandmaster_fen, grandmaster_eval, db_conn)

            if grandmaster_eval is not None:
                legal_moves_data.append({
                    "move": move.uci(),
                    "eval_type": grandmaster_eval["eval_type"],
                    "value": grandmaster_eval["value"] / 100 if grandmaster_eval["eval_type"] == "cp" else grandmaster_eval["value"]
                })

        # Make the move on the board
        board.push(move)

        # Evaluate the board position after the grandmaster's move
        fen_after_move = normalize_fen(board.fen())
        eval_data = fetch_evaluation(fen_after_move, db_conn)

        if eval_data is None:
            # Evaluate the FEN after the move using Stockfish with limited depth
            eval_data = evaluate_with_stockfish(fen_after_move)
            if eval_data:
                stockfish_eval_count += 1  # Increment Stockfish evaluation counter
                save_to_database(fen_after_move, eval_data, db_conn)
        else:
            db_eval_count += 1  # Increment database evaluation counter

        # Add the training data entry
        if eval_data is not None:
            value = eval_data["value"] / 100 if eval_data["eval_type"] == "cp" else eval_data["value"]
            training_data.append({
                "fen": fen_before_move,  # FEN before the move
                "moves": moves[:-1],     # Moves leading up to the current FEN
                "next_move": move.uci(), # Move played in the game
                "value": value,          # Evaluation of the FEN after the grandmaster's move
                "legal_moves": legal_moves_data  # Evaluations of all legal moves before the move
            })

    return training_data

"""def process_game(game, db_conn):
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
            # Determine the evaluation value to store based on the eval data
            if eval_data["eval_type"] == "cp":
                value = eval_data["value"] / 100  # Convert centipawns to pawns
            else:  # Mate in n
                value = eval_data["value"]

            # Add the training data entry
            training_data.append({
                "fen": fen_before_move,  # FEN before the move
                "moves": moves[:-1],     # Moves leading up to the current FEN
                "next_move": move.uci(), # Move played in the game
                "value": value           # Evaluation of the FEN after the move
            })

    return training_data"""


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

    #print("Counting the total PGN games...")
    #total_games = count_games_in_pgn(pgn_path)
    #print(f"Total games in PGN: {total_games}")

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
