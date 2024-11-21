import chess.pgn
import pandas as pd
import os  # To handle directory creation

def pgn_to_fen_by_elo(pgn_file, output_dir, max_games=1000):
    ranges = {
        "less_1000": (0, 1000),
        "1000_1500": (1000, 1500),
        "1500_2000": (1500, 2000),
        "greater_2000": (2000, float("inf"))
    }
    data = {key: [] for key in ranges.keys()}
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(pgn_file, "r") as f:
        for i, game in enumerate(iter(lambda: chess.pgn.read_game(f), None)):
            if i >= max_games:  # Stop after processing max_games
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
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        data[key].append((board.fen(), move.uci(), white_elo, black_elo))
                    break  # A game only goes into one range
    
    # Save each range to a separate CSV
    for key, rows in data.items():
        df = pd.DataFrame(rows, columns=["FEN", "Move", "WhiteElo", "BlackElo"])
        output_file = f"{output_dir}/{key}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(rows)} moves to {output_file}")

# Call the function
pgn_to_fen_by_elo("lichess_games.pgn", "output_data", max_games=1000)
