import chess.pgn

# Function to count games by ELO range
def count_games_by_elo(pgn_file, elo_ranges):
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
                avg_elo = (int(headers["WhiteElo"]) + int(headers["BlackElo"])) // 2
                matched = False
                for range_ in elo_ranges:
                    if range_[0] <= avg_elo <= range_[1]:
                        elo_counts[f"{range_[0]}-{range_[1]}"] += 1
                        matched = True
                        break
                if not matched:
                    elo_counts["Unknown"] += 1
            else:
                elo_counts["Unknown"] += 1

            if game_index % 10000 == 0:  # Progress update every 1000 games
                print(f"Processed {game_index} games...")

    return elo_counts

# Define ELO ranges
elo_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 4000)]

# Path to the PGN file
pgn_file = "./partial_lichess_games_50k_filtered_elo.pgn"

# Count games by ELO range
print("Counting games by ELO range...")
elo_counts = count_games_by_elo(pgn_file, elo_ranges)

# Print results
print("\nELO Range Counts:")
for elo_range, count in elo_counts.items():
    print(f"{elo_range}: {count}")
