import chess.pgn

def filter_games_by_termination_and_moves(input_pgn, output_pgn, valid_terminations, min_moves=2):
    """
    Filters games in a PGN file by the Termination field and a minimum number of moves.

    Args:
        input_pgn (str): Path to the input PGN file.
        output_pgn (str): Path to the output PGN file.
        valid_terminations (list): List of Termination strings to include.
        min_moves (int): Minimum number of moves for a game to be included.
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

            # Count the number of moves in the game
            moves = list(game.mainline_moves())
            move_count = len(moves)

            if termination in valid_terminations or (
                termination == "Time forfeit" and move_count >= min_moves
            ):
                filtered_count += 1
                outfile.write(str(game) + "\n\n")
            else:
                # Optionally log skipped games
                print(
                    f"Skipping game {game_count}: Termination='{termination}', Moves={move_count}"
                )

        print(f"Total games processed: {game_count}")
        print(f"Games written to output: {filtered_count}")

# Example usage
input_pgn_path = "../../PGN Games/partial_lichess_games_15k.pgn"
output_pgn_path = "../../PGN Games/partial_lichess_games_15k_filtered.pgn"
valid_termination_conditions = [
    "Normal",           # Includes games that ended normally
    "Checkmate",        # Games that ended in checkmate
    "Stalemate",        # Games that ended in stalemate
    "Insufficient material",  # Games that ended due to insufficient material
    "Threefold repetition",   # Games that ended due to threefold repetition
    "50-move rule"      # Games that ended due to the 50-move rule
]

# Include games with "Time forfeit" if they have at least 2 moves
filter_games_by_termination_and_moves(input_pgn_path, output_pgn_path, valid_termination_conditions, min_moves=2)

