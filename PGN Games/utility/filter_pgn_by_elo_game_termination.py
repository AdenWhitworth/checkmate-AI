"""
Script to filter chess games from a PGN file based on termination conditions, ELO ratings, and move count.

This script:
1. Reads games from an input PGN file.
2. Filters games based on termination conditions, minimum ELO, and minimum move count.
3. Writes the filtered games to an output PGN file.

Functions:
- filter_games_by_termination_elo_and_moves: Filters games based on specified criteria.

Usage:
- Update the paths for `input_pgn_path` and `output_pgn_path`.
- Define valid termination conditions, minimum ELO, and move count.
- Run the script to filter games and write the results to the output file.

Requirements:
- python-chess library
"""

import chess.pgn


def filter_games_by_termination_elo_and_moves(input_pgn, output_pgn, valid_terminations, min_moves=2, min_elo=2000):
    """
    Filters chess games from a PGN file based on termination conditions, ELO, and move count.

    Args:
        input_pgn (str): Path to the input PGN file.
        output_pgn (str): Path to the output PGN file to save filtered games.
        valid_terminations (list of str): List of acceptable termination conditions.
        min_moves (int): Minimum number of moves required for a game (default is 2).
        min_elo (int): Minimum ELO rating for both players (default is 2000).

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
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            move_count = len(list(game.mainline_moves()))  # Count moves in the game

            # Filtering criteria
            if (
                white_elo >= min_elo and black_elo >= min_elo
                and (termination in valid_terminations or (termination == "Time forfeit" and move_count >= min_moves))
            ):
                filtered_count += 1
                outfile.write(str(game) + "\n\n")

                # Print update every 5,000 filtered games
                if filtered_count % 5000 == 0:
                    print(f"Filtered {filtered_count} games out of {game_count} processed.")

        # Final summary
        print(f"Total games processed: {game_count}")
        print(f"Games written to output: {filtered_count}")


if __name__ == "__main__":
    #Set filter conditions
    min_elo = 2000
    min_moves = 2

    # Input and output file paths
    input_pgn_path = "../../PGN Games/pgns/partial_lichess_games_150k.pgn"
    output_pgn_path = "../../PGN Games/pgns/partial_lichess_games_150k_filtered_elo.pgn"

    # Termination conditions to include
    valid_termination_conditions = [
        "Normal",               # Games ended normally
        "Checkmate",            # Games ended in checkmate
        "Stalemate",            # Games ended in stalemate
        "Insufficient material",  # Games ended due to insufficient material
        "Threefold repetition", # Games ended due to threefold repetition
        "50-move rule"          # Games ended due to the 50-move rule
    ]

    # Include games with "Time forfeit" if they have at least 2 moves
    filter_games_by_termination_elo_and_moves(
        input_pgn_path,
        output_pgn_path,
        valid_termination_conditions,
        min_moves,
        min_elo
    )
