"""
Script to filter chess games in a PGN file based on termination conditions and move count.

This script:
1. Reads games from an input PGN file.
2. Filters games based on their Termination header and minimum number of moves.
3. Writes the filtered games to an output PGN file.

Functions:
- filter_games_by_termination_and_moves: Filters games from a PGN file based on criteria.

Usage:
- Update `input_pgn_path` and `output_pgn_path` as needed.
- Define valid termination conditions and minimum move requirements.
- Run the script to filter the games.

Requirements:
- python-chess library
"""

import chess.pgn


def filter_games_by_termination_and_moves(input_pgn, output_pgn, valid_terminations, min_moves=2, log_skipped=False):
    """
    Filters games in a PGN file by the Termination field and a minimum number of moves.

    Args:
        input_pgn (str): Path to the input PGN file.
        output_pgn (str): Path to the output PGN file.
        valid_terminations (list): List of valid Termination strings to include.
        min_moves (int): Minimum number of moves required for a game to be included (default is 2).
        log_skipped (bool): Whether to log skipped games for debugging purposes (default is False).

    Returns:
        None
    """
    with open(input_pgn, "r") as infile, open(output_pgn, "w") as outfile:
        game_count = 0
        filtered_count = 0

        while True:
            game = chess.pgn.read_game(infile)
            if game is None:
                break

            game_count += 1
            termination = game.headers.get("Termination", None)
            move_count = len(list(game.mainline_moves()))  # Count moves in the game

            # Check if the game meets filtering criteria
            if termination in valid_terminations or (
                termination == "Time forfeit" and move_count >= min_moves
            ):
                filtered_count += 1
                outfile.write(str(game) + "\n\n")
            elif log_skipped:
                # Optionally log skipped games for debugging
                print(
                    f"Skipping game {game_count}: Termination='{termination}', Moves={move_count}"
                )

        # Summary
        print(f"Total games processed: {game_count}")
        print(f"Games written to output: {filtered_count}")


if __name__ == "__main__":
    # Input and output file paths
    input_pgn_path = "../../PGN Games/pgns/partial_lichess_games_15k.pgn"
    output_pgn_path = "../../PGN Games/pgns/partial_lichess_games_15k_filtered.pgn"

    # Valid termination conditions
    valid_termination_conditions = [
        "Normal",               # Games ended normally
        "Checkmate",            # Games ended in checkmate
        "Stalemate",            # Games ended in stalemate
        "Insufficient material",  # Games ended due to insufficient material
        "Threefold repetition", # Games ended due to threefold repetition
        "50-move rule"          # Games ended due to the 50-move rule
    ]

    # Filter games with at least 2 moves and valid terminations
    filter_games_by_termination_and_moves(
        input_pgn_path,
        output_pgn_path,
        valid_termination_conditions,
        min_moves=2,
        log_skipped=True  # Enable skipped game logging for debugging
    )

