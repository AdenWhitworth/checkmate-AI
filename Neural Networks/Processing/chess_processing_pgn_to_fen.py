import chess.pgn
import pandas as pd
import os
import chess

def validate_all_rows(csv_file):
    """
    Validates all rows in a CSV file containing FEN positions and moves.
    Ensures that the pgn move made from the initial state fen is a valid
    chess move.

    Args:
        csv_file (str): Path to the CSV file to validate.

    Returns:
        list: A list of errors, where each error is a tuple (index, FEN, Move, Error).
    """
    df = pd.read_csv(csv_file)
    errors = []
    
    if df.empty:
        print(f"The CSV file {csv_file} is empty.")
        return errors

    print(f"Validating all rows in the file: {csv_file}")

    for index, row in df.iterrows():
        try:
            fen = row["FEN"]
            move = row["Move"]
            
            # Initialize the board with the FEN
            board = chess.Board(fen)

            # Convert move to a chess.Move object
            move_obj = chess.Move.from_uci(move)

            # Check if the move is legal
            if move_obj not in board.legal_moves:
                raise ValueError("Move not legal")
            
        except Exception as e:
            errors.append((index, row["FEN"], row["Move"], str(e)))
            print(f"Error in Row {index}: FEN={row['FEN']}, Move={row['Move']}, Error={e}")
    
    return errors


def pgn_to_fen_all_elo(pgn_file, output_dir, max_games_per_range=15000):
    """
    Converts PGN games into FEN positions grouped by ELO ranges and validates the output.

    Args:
        pgn_file (str): Path to the input PGN file.
        output_dir (str): Directory to save the output CSV files.
        max_games_per_range (int): Maximum number of games per ELO range.
    """
    ranges = {
        "less_1000": (0, 1000),
        "1000_1500": (1000, 1500),
        "1500_2000": (1500, 2000),
        "greater_2000": (2000, float("inf"))
    }
    data = {key: [] for key in ranges.keys()}
    game_count = {key: 0 for key in ranges.keys()}  # Track games per range

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(pgn_file, "r") as f:
        for i, game in enumerate(iter(lambda: chess.pgn.read_game(f), None)):
            # Stop if all ranges have enough games
            if all(count >= max_games_per_range for count in game_count.values()):
                break
            
            # Get ELO ratings from headers
            white_elo = game.headers.get("WhiteElo", "Unknown")
            black_elo = game.headers.get("BlackElo", "Unknown")
            
            try:
                white_elo = int(white_elo)
                black_elo = int(black_elo)
            except ValueError:
                continue  # Skip games with invalid ELOs
            
            # Check which range this game belongs to
            avg_elo = (white_elo + black_elo) // 2
            for key, (low, high) in ranges.items():
                if low <= avg_elo < high and game_count[key] < max_games_per_range:
                    board = chess.Board()
                    for move in game.mainline_moves():
                        fen_before_move = board.fen()  # FEN before the move is applied
                        move_uci = move.uci()  # Convert move to UCI
                        
                        if move not in board.legal_moves:
                            print(f"Skipping invalid move: {move_uci} in game {i}")
                            continue
                        
                        board.push(move)  # Apply the move to the board
                        
                        data[key].append((fen_before_move, move_uci, white_elo, black_elo))
                    
                    game_count[key] += 1  # Increment the game count for this range
                    break  # A game only goes into one range
    
    # Save each range to a separate CSV and validate it
    for key, rows in data.items():
        df = pd.DataFrame(rows, columns=["FEN", "Move", "WhiteElo", "BlackElo"])
        output_file = f"{output_dir}/{key}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(rows)} moves from {game_count[key]} games to {output_file}")

        # Validate the created file
        errors = validate_all_rows(output_file)
        if errors:
            print(f"\nValidation errors found in {output_file}:")
            for error in errors:
                print(f"Row {error[0]}: FEN={error[1]}, Move={error[2]}, Error={error[3]}")
        else:
            print(f"No validation errors found in {output_file}.")


def pgn_to_fen_base_elos(pgn_file, output_dir, max_games_per_range=15000):
    """
    Extracts games from the 1500-2000 and greater than 2000 ELO ranges from a PGN file,
    converts them to FEN format, and saves to a single CSV file in the specified directory.

    Args:
        pgn_file (str): Path to the input PGN file.
        output_dir (str): Directory where the output CSV file should be placed.
        max_games_per_range (int): Maximum number of games to process per range.
    """
    combined_data = []
    combined_game_count = 0
    ranges = {"1500_2000": (1500, 2000), "greater_2000": (2000, float("inf"))}

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(pgn_file, "r") as f:
        for i, game in enumerate(iter(lambda: chess.pgn.read_game(f), None)):
            # Stop if the combined game count exceeds the max_games_per_range
            if combined_game_count >= max_games_per_range:
                break

            # Get ELO ratings from headers
            white_elo = game.headers.get("WhiteElo", "Unknown")
            black_elo = game.headers.get("BlackElo", "Unknown")

            try:
                white_elo = int(white_elo)
                black_elo = int(black_elo)
            except ValueError:
                continue  # Skip games with invalid ELOs

            # Check which range this game belongs to
            avg_elo = (white_elo + black_elo) // 2
            for key, (low, high) in ranges.items():
                if low <= avg_elo < high:
                    board = chess.Board()
                    for move in game.mainline_moves():
                        fen_before_move = board.fen()  # FEN before the move is applied
                        move_uci = move.uci()  # Convert move to UCI
                        
                        if move not in board.legal_moves:
                            print(f"Skipping invalid move: {move_uci} in game {i}")
                            continue

                        board.push(move)  # Apply the move to the board
                        combined_data.append((fen_before_move, move_uci, white_elo, black_elo))
                    
                    combined_game_count += 1  # Increment the combined game count
                    break  # Move to the next game after processing

    # Define the output CSV path
    output_csv = os.path.join(output_dir, "base_games_fen.csv")

    # Save the combined data to a single CSV
    df = pd.DataFrame(combined_data, columns=["FEN", "Move", "WhiteElo", "BlackElo"])
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(combined_data)} moves from {combined_game_count} games to {output_csv}")

    # Validate the created file
    errors = validate_all_rows(output_csv)
    if errors:
        print(f"\nValidation errors found in {output_csv}:")
        for error in errors:
            print(f"Row {error[0]}: FEN={error[1]}, Move={error[2]}, Error={error[3]}")
    else:
        print(f"No validation errors found in {output_csv}.")


# Run the processing function for all ELO ranges for fine tuning
#pgn_to_fen_all_elo("../../PGN Games/partial_lichess_games_150k.pgn", "5000_GAMES_FENS", max_games_per_range=5000)

# Run the processing for base ELO ranges for base training. Ensure different games are processed.
pgn_to_fen_base_elos("../../PGN Games/partial_lichess_games_150k-300k.pgn", "5000_GAMES_FENS", max_games_per_range=15000)







