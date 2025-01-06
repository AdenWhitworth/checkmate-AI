"""
Script to count chess games by ELO range from a PGN file.

This script:
1. Reads chess games from a PGN file.
2. Counts the number of games falling within specified ELO ranges.
3. Handles games without ELO information and counts them separately.

Functions:
- count_games_by_elo: Counts games by ELO range and unknown ELO.
- print_elo_counts: Displays the counts of games for each ELO range.

Requirements:
- python-chess library

Usage:
- Update the `elo_ranges` and `pgn_file` paths as needed.
- Run the script to see the count of games by ELO range.
"""

import chess.pgn


def count_games_by_elo(pgn_file, elo_ranges):
    """
    Counts chess games by ELO range from a PGN file.

    Args:
        pgn_file (str): Path to the PGN file containing chess games.
        elo_ranges (list of tuple): List of ELO ranges as (min, max) pairs.

    Returns:
        dict: A dictionary where keys are ELO range strings (e.g., '1000-1500')
              or 'Unknown' and values are the count of games in that range.
    """
    elo_counts = {f"{range_[0]}-{range_[1]}": 0 for range_ in elo_ranges}
    elo_counts["Unknown"] = 0  # For games without ELO information

    with open(pgn_file, "r") as file:
        game_index = 0
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break

            game_index += 1
            headers = game.headers
            if "WhiteElo" in headers and "BlackElo" in headers:
                try:
                    avg_elo = (int(headers["WhiteElo"]) + int(headers["BlackElo"])) // 2
                    matched = False
                    for range_ in elo_ranges:
                        if range_[0] <= avg_elo <= range_[1]:
                            elo_counts[f"{range_[0]}-{range_[1]}"] += 1
                            matched = True
                            break
                    if not matched:
                        elo_counts["Unknown"] += 1
                except ValueError:
                    # Handles cases where ELO values are not integers
                    elo_counts["Unknown"] += 1
            else:
                elo_counts["Unknown"] += 1

            if game_index % 10000 == 0:  # Progress update every 10,000 games
                print(f"Processed {game_index} games...")

    return elo_counts


def print_elo_counts(elo_counts):
    """
    Prints the counts of games for each ELO range.

    Args:
        elo_counts (dict): Dictionary containing ELO range counts.
    """
    print("\nELO Range Counts:")
    for elo_range, count in elo_counts.items():
        print(f"{elo_range}: {count}")


if __name__ == "__main__":
    # Define ELO ranges
    elo_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 4000)]

    # Path to the PGN file
    pgn_file = "../pgns/partial_lichess_games_50k_filtered_elo.pgn"

    # Count games by ELO range
    print("Counting games by ELO range...")
    elo_counts = count_games_by_elo(pgn_file, elo_ranges)

    # Print results
    print_elo_counts(elo_counts)
