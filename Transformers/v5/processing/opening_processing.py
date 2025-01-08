"""
Chess PGN Processing Script

This script processes chess games stored in PGN (Portable Game Notation) format to identify
openings and match them with a predefined openings database. It extracts key information such as 
move sequences, board states (FENs), and game outcomes, and saves the processed data in JSON format.

### Key Features:
1. **Openings Matching**:
    - Uses a CSV file containing chess openings to match games with their openings based on move sequences.
    - Extracts metadata such as ECO code, opening name, and moves.

2. **PGN Parsing**:
    - Parses chess games from a PGN file.
    - Extracts moves, FENs, and outcomes for each game.

3. **Game Outcome Extraction**:
    - Determines the result of each game: 
        - `1` for White win, `-1` for Black win, and `0` for a draw.

4. **JSON Output**:
    - Saves processed game data to a JSON file for further analysis or training machine learning models.

### Functions:
- **load_openings_csv**: Loads and validates the openings database from a CSV file.
- **count_games_in_pgn**: Estimates the number of games in a PGN file.
- **parse_game_moves**: Extracts moves, FENs, and outcomes for a single game while matching with the openings database.
- **process_pgn_with_openings**: Processes multiple games in a PGN file, matching openings and extracting key details.
- **save_to_json**: Saves the processed data to a JSON file.
- **main**: Orchestrates the loading, processing, and saving workflow.

### Usage:
1. Update the file paths for:
    - `OPENINGS_CSV_PATH`: Path to the openings CSV file.
    - `PGN_PATH`: Path to the PGN file.
    - `OUTPUT_JSON_PATH`: Path to save the output JSON file.
2. Run the script:
    ```
    python script_name.py
    ```

### Output:
- A JSON file containing processed games with matched openings and outcomes.

"""
import pandas as pd
import chess.pgn
import json
from tqdm import tqdm

def load_openings_csv(file_path):
    """
    Load the openings CSV file and validate required columns.

    Args:
        file_path (str): Path to the openings CSV file.

    Returns:
        pd.DataFrame: Dataframe containing the openings data.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """
    openings_df = pd.read_csv(file_path)
    required_columns = ["eco", "name", "uci", "epd"]
    for column in required_columns:
        if column not in openings_df.columns:
            raise ValueError(f"Missing required column '{column}' in openings CSV.")
    return openings_df

def count_games_in_pgn(pgn_path):
    """
    Estimate the number of games in a PGN file.

    Args:
        pgn_path (str): Path to the PGN file.

    Returns:
        int: Total number of games in the PGN file.
    """
    with open(pgn_path, "r") as pgn:
        return sum(1 for _ in chess.pgn.read_game(pgn))

def parse_game_moves(game, openings_df):
    """
    Parse the moves and FENs for a single game, matching against the openings database.

    Args:
        game (chess.pgn.Game): PGN game object.
        openings_df (pd.DataFrame): Dataframe containing openings data.

    Returns:
        dict: Processed game data with opening information and outcome, or None if no opening match is found.
    """
    board = game.board()
    move_list_uci = []
    fens = []
    found_opening = None

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

    if found_opening:
        # Extract game outcome
        result = game.headers.get("Result", "")
        if result == "1-0":
            outcome = 1  # White wins
        elif result == "0-1":
            outcome = -1  # Black wins
        elif result == "1/2-1/2":
            outcome = 0  # Draw
        else:
            outcome = None  # Unknown

        return {
            "eco": found_opening["eco"],
            "opening_name": found_opening["name"],
            "moves": move_list_uci,
            "fens": fens,
            "game_outcome": outcome
        }
    return None

def process_pgn_with_openings(pgn_path, openings_df):
    """
    Process a PGN file and match games against an openings database.

    Args:
        pgn_path (str): Path to the PGN file.
        openings_df (pd.DataFrame): Dataframe containing openings data.

    Returns:
        list: List of processed games with matched openings and outcomes.
    """
    processed_games = []
    with open(pgn_path, "r") as pgn:
        for _ in tqdm(range(26173), desc="Processing PGN games"):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            processed_game = parse_game_moves(game, openings_df)
            if processed_game:
                processed_games.append(processed_game)
    return processed_games

def save_to_json(data, output_path):
    """
    Save data to a JSON file.

    Args:
        data (list): Data to save.
        output_path (str): Path to the output JSON file.
    """
    with open(output_path, "w") as output_file:
        json.dump(data, output_file, indent=2)

def main(OPENINGS_CSV_PATH, PGN_PATH, OUTPUT_JSON_PATH):
    """
    Main function to process PGN games and save matched openings and outcomes.
    Args:
        OPENINGS_CSV_PATH (str): Path to the openings CSV file.
        PGN_PATH (str): Path to the PGN file.
        OUTPUT_JSON_PATH (str): Path to the output JSON file.
    """
    print("Loading openings data...")
    openings_df = load_openings_csv(OPENINGS_CSV_PATH)

    print("Processing games...")
    processed_games = process_pgn_with_openings(PGN_PATH, openings_df)
    print(f"Processed {len(processed_games)} games.")

    print(f"Saving results to {OUTPUT_JSON_PATH}...")
    save_to_json(processed_games, OUTPUT_JSON_PATH)
    print(f"Results saved to {OUTPUT_JSON_PATH}.")

if __name__ == "__main__":
    OPENINGS_CSV_PATH = r"D:\checkmate_ai\game_phases\openings.csv"
    PGN_PATH = "../../PGN Games/partial_lichess_games_26k_filtered_2000_elo.pgn"
    OUTPUT_JSON_PATH = r"D:\checkmate_ai\game_phases\open_data.json"

    main(OPENINGS_CSV_PATH, PGN_PATH, OUTPUT_JSON_PATH)
