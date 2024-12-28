import pandas as pd
import chess.pgn
import json
from tqdm import tqdm
import io

# File paths
openings_csv_path = r"D:\checkmate_ai\game_phases\openings.csv"
pgn_path = "../../PGN Games/partial_lichess_games_26k_filtered_2000_elo.pgn"
output_json_path = r"D:\checkmate_ai\game_phases\open_data.json"

# Load openings CSV
openings_df = pd.read_csv(openings_csv_path)

# Ensure required columns exist in the openings CSV
required_columns = ["eco", "name", "uci", "epd"]
for column in required_columns:
    if column not in openings_df.columns:
        raise ValueError(f"Missing required column '{column}' in openings CSV.")

# Function to estimate the number of games in the PGN file
def count_games_in_pgn(pgn_path):
    with open(pgn_path, "r") as pgn:
        return sum(1 for _ in chess.pgn.read_game(pgn))

# Parse PGN games and generate output JSON
def process_pgn_with_openings(pgn_path, openings_df):
    processed_games = []
    #total_games = count_games_in_pgn(pgn_path)  # Count total games for progress bar

    with open(pgn_path, "r") as pgn:
        for game_number in tqdm(range(26173), desc="Processing PGN games"):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()
            move_list_uci = []
            fens = []
            found_opening = None

            # Parse moves and FENs
            for move in game.mainline_moves():
                move_list_uci.append(move.uci())
                board.push(move)
                fens.append(board.fen())

                # Check for opening match
                current_moves = " ".join(move_list_uci)
                match = openings_df[openings_df["uci"].str.startswith(current_moves)]
                if not match.empty:
                    found_opening = match.iloc[0]
                else:
                    break

            if found_opening is not None:
                # Extract game outcome
                result = game.headers["Result"]
                if result == "1-0":
                    outcome = 1  # White wins
                elif result == "0-1":
                    outcome = -1  # Black wins
                elif result == "1/2-1/2":
                    outcome = 0  # Draw
                else:
                    outcome = None  # Unknown

                # Create output entry
                processed_games.append({
                    "eco": found_opening["eco"],
                    "opening_name": found_opening["name"],
                    "moves": move_list_uci,
                    "fens": fens,
                    "game_outcome": outcome
                })

    return processed_games

# Process games and save to JSON
print("Processing games...")
processed_games = process_pgn_with_openings(pgn_path, openings_df)
print(f"Processed {len(processed_games)} games.")

# Save output
with open(output_json_path, "w") as output_file:
    json.dump(processed_games, output_file, indent=2)

print(f"Saved output to {output_json_path}")
