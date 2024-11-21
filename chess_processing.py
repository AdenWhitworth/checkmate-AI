import chess.pgn
import pandas as pd

def pgn_to_fen(pgn_file, output_csv, max_games=1000):
    games = []
    with open(pgn_file, "r") as f:
        for i, game in enumerate(iter(lambda: chess.pgn.read_game(f), None)):
            if i >= max_games:  # Stop after 1000 games
                break

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                # Collect FEN, move, and player ratings
                white_elo = game.headers.get("WhiteElo", "Unknown")
                black_elo = game.headers.get("BlackElo", "Unknown")
                games.append((board.fen(), move.uci(), white_elo, black_elo))
    
    # Save to CSV
    df = pd.DataFrame(games, columns=["FEN", "Move", "WhiteElo", "BlackElo"])
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(games)} board states to {output_csv}")

# Call the function
pgn_to_fen("lichess_games.pgn", "output.csv", max_games=1000)

