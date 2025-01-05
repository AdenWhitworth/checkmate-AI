import json
from collections import defaultdict
import chess.pgn

def extract_games(pgn_file):
    """Extract all games from a PGN file."""
    games = []
    with open(pgn_file, "r") as file:
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break
            moves = []
            board = game.board()
            for move in game.mainline_moves():
                moves.append(board.san(move))
                board.push(move)
            games.append(moves)
    return games

# Load the dataset (replace with your actual dataset file path)
pgn_file = "../PGN Games/partial_lichess_games_15k.pgn"
all_games = extract_games(pgn_file)

# Build the vocabulary
vocab = defaultdict(int)
for game in all_games:
    for move in game:
        vocab[move] += 1

move_to_id = {move: idx for idx, move in enumerate(vocab.keys())}
id_to_move = {idx: move for move, idx in move_to_id.items()}

# Save mappings to JSON files
with open("models/base_transformer_15k_games_Models/move_to_id.json", "w") as f:
    json.dump(move_to_id, f)

with open("models/base_transformer_15k_games_Models/id_to_move.json", "w") as f:
    json.dump(id_to_move, f)

print("Mappings saved: move_to_id.json and id_to_move.json")
