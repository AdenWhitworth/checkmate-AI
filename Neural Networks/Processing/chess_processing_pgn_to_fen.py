"""
Script to process PGN chess games, convert them to FEN format, and group them by ELO ranges.

This script provides functionality to:
1. Convert PGN files to FEN positions grouped by ELO ranges.
2. Validate the generated FEN positions and moves for correctness.

Functions:
- validate_all_rows: Validates rows in a CSV containing FEN positions and moves.
- pgn_to_fen_all_elo: Processes PGN games for all ELO ranges and saves them to separate CSVs.
- pgn_to_fen_base_elos: Processes games in selected ELO ranges and saves them to a combined CSV.

Requirements:
- python-chess
- pandas

Usage:
- Modify the PGN file paths and output directories as needed, then run the desired processing function.
"""

import chess.pgn
import pandas as pd
import os
import chess


def validate_all_rows(csv_file):
    """
    Validates all rows in a CSV file containing FEN positions and moves.
    Ensures that the move made from the initial FEN state is a valid chess move.

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


def process_pgn_games(pgn_file, output_dir, ranges, max_games_per_range, combined=False):
    """
    Generic function to process PGN games, convert them to FEN, and save grouped by ELO range.

    Args:
        pgn_file (str): Path to the input PGN file.
        output_dir (str): Directory to save the output CSV files.
        ranges (dict): ELO ranges as a dictionary (e.g., {"1500_2000": (1500, 2000)}).
        max_games_per_range (int): Maximum number of games to process per range.
        combined (bool): If True, saves all ranges to a single CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = {key: [] for key in ranges.keys()}
    game_count = {key: 0 for key in ranges.keys()}  # Track games per range
    combined_data = []

    with open(pgn_file, "r") as f:
        for i, game in enumerate(iter(lambda: chess.pgn.read_game(f), None)):
            # Stop if all ranges (or combined) have enough games
            if combined and len(combined_data) >= max_games_per_range:
                break
            if not combined and all(count >= max_games_per_range for count in game_count.values()):
                break

            # Get ELO ratings from headers
            white_elo = game.headers.get("WhiteElo", "Unknown")
            black_elo = game.headers.get("BlackElo", "Unknown")

            try:
                white_elo = int(white_elo)
                black_elo = int(black_elo)
            except ValueError:
                continue  # Skip games with invalid ELOs

            avg_elo = (white_elo + black_elo) // 2

            for key, (low, high) in ranges.items():
                if low <= avg_elo < high and (combined or game_count[key] < max_games_per_range):
                    board = chess.Board()
                    for move in game.mainline_moves():
                        fen_before_move = board.fen()
                        move_uci = move.uci()

                        if move not in board.legal_moves:
                            print(f"Skipping invalid move: {move_uci} in game {i}")
                            continue

                        board.push(move)
                        row = (fen_before_move, move_uci, white_elo, black_elo)

                        if combined:
                            combined_data.append(row)
                        else:
                            data[key].append(row)

                    if not combined:
                        game_count[key] += 1
                    break

    # Save the results
    if combined:
        output_csv = os.path.join(output_dir, "combined_games_fen.csv")
        save_to_csv(combined_data, output_csv)
    else:
        for key, rows in data.items():
            output_csv = os.path.join(output_dir, f"{key}.csv")
            save_to_csv(rows, output_csv)


def save_to_csv(data, output_csv):
    """
    Saves the provided data to a CSV file and validates it.

    Args:
        data (list): Data to save as a list of tuples (FEN, Move, WhiteElo, BlackElo).
        output_csv (str): Path to save the CSV file.
    """
    df = pd.DataFrame(data, columns=["FEN", "Move", "WhiteElo", "BlackElo"])
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(data)} rows to {output_csv}")

    # Validate the CSV
    errors = validate_all_rows(output_csv)
    if errors:
        print(f"Validation errors found in {output_csv}:")
        for error in errors:
            print(f"Row {error[0]}: FEN={error[1]}, Move={error[2]}, Error={error[3]}")
    else:
        print(f"No validation errors found in {output_csv}.")

if __name__ == "__main__":
    ranges_all = {
        "less_1000": (0, 1000),
        "1000_1500": (1000, 1500),
        "1500_2000": (1500, 2000),
        "greater_2000": (2000, float("inf"))
    }

    ranges_base = {
        "1500_2000": (1500, 2000),
        "greater_2000": (2000, float("inf"))
    }

    # Process 5000 games for all ELO ranges
    process_pgn_games("../../PGN Games/pgns/partial_lichess_games_300k.pgn", "5000_GAMES_FENS", ranges_all, max_games_per_range=5000)

    # Process games for base ELO ranges in combined mode
    process_pgn_games("../../PGN Games/pgns/partial_lichess_games_300k.pgn", "15000_GAMES_FENS", ranges_base, max_games_per_range=15000, combined=True)
