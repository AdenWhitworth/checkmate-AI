import chess
import chess.pgn
import sqlite3

def check_fens_in_database(game, db_path):
    """
    Check if FENs from a single game are present in the database and print results.
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

def normalize_fen(fen):
    """
    Normalize a FEN string by keeping only the first four fields.
    """
    return " ".join(fen.split()[:4])

# Example usage
db_path = r"D:\checkmate_ai\evaluations.db"
pgn_path = "../../PGN Games/partial_lichess_games_26k_filtered_2000_elo.pgn"

with open(pgn_path) as pgn:
    game = chess.pgn.read_game(pgn)  # Read the first game
    if game:
        check_fens_in_database(game, db_path)

