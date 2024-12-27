import chess.pgn

def filter_games_by_termination_elo_and_moves(input_pgn, output_pgn, valid_terminations, min_moves=2, min_elo=2000):
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

            # Count the number of moves in the game
            moves = list(game.mainline_moves())
            move_count = len(moves)

            # Check if the game meets the filtering criteria
            if (
                white_elo >= min_elo and black_elo >= min_elo  # Both players have Elo >= min_elo
                and (termination in valid_terminations or (termination == "Time forfeit" and move_count >= min_moves))
            ):
                filtered_count += 1
                outfile.write(str(game) + "\n\n")

                # Print update every 5,000 games filtered
                if filtered_count % 5000 == 0:
                    print(f"Filtered {filtered_count} games out of {game_count} processed.")

        # Final summary
        print(f"Total games processed: {game_count}")
        print(f"Games written to output: {filtered_count}")

# Example usage
input_pgn_path = "../../PGN Games/partial_lichess_games_150k.pgn"
output_pgn_path = "../../PGN Games/partial_lichess_games_150k_filtered_elo.pgn"
valid_termination_conditions = [
    "Normal",           # Includes games that ended normally
    "Checkmate",        # Games that ended in checkmate
    "Stalemate",        # Games that ended in stalemate
    "Insufficient material",  # Games that ended due to insufficient material
    "Threefold repetition",   # Games that ended due to threefold repetition
    "50-move rule"      # Games that ended due to the 50-move rule
]

# Include games with "Time forfeit" if they have at least 2 moves
filter_games_by_termination_elo_and_moves(input_pgn_path, output_pgn_path, valid_termination_conditions, min_moves=2, min_elo=2000)