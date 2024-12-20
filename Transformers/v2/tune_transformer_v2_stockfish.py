import chess.pgn
import chess.engine
import numpy as np
import json
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function to extract games by ELO range
def extract_games_by_elo(pgn_file, elo_range, max_games):
    with open(pgn_file, "r") as file:
        games = []
        while len(games) < max_games:
            game = chess.pgn.read_game(file)
            if game is None:
                break
            headers = game.headers
            if "WhiteElo" in headers and "BlackElo" in headers:
                avg_elo = (int(headers["WhiteElo"]) + int(headers["BlackElo"])) // 2
                if elo_range[0] <= avg_elo <= elo_range[1]:
                    moves = []
                    board = game.board()
                    for move in game.mainline_moves():
                        moves.append(board.san(move))
                        board.push(move)
                    games.append(moves)
    return games

# Annotate game phase
def annotate_game_phase_for_partial(split_point):
    if split_point < 10:
        return "opening"
    elif split_point < 30:
        return "mid-game"
    else:
        return "end-game"

# Convert board to tensor
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = (piece.piece_type - 1) + (6 if piece.color else 0)
            row, col = chess.square_rank(square), chess.square_file(square)
            tensor[row, col, layer] = 1
    return tensor.flatten()

# Generate partial games with phase labels
def generate_partial_games(games, move_to_id, num_partial_per_game=3):
    partial_inputs, partial_outputs, partial_boards, partial_phases = [], [], [], []
    for game in games:
        game_length = len(game)
        for _ in range(num_partial_per_game):
            split_point = np.random.randint(1, game_length)
            phase = annotate_game_phase_for_partial(split_point)
            partial_game = game[:split_point]
            inputs, outputs, boards, phases = process_games_with_board_features([partial_game], move_to_id)
            partial_inputs.extend(inputs)
            partial_outputs.extend(outputs)
            partial_boards.extend(boards)
            partial_phases.extend([phase] * len(inputs))
    return partial_inputs, partial_outputs, partial_boards, partial_phases

# Annotate games with Stockfish
def annotate_with_stockfish(games, stockfish_path, depth=10):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    annotated_games = []
    try:
        for idx, game in enumerate(games):
            board = chess.Board()
            annotated_game = []
            for move in game:
                board.push_san(move)
                result = engine.analyse(board, chess.engine.Limit(depth=depth))
                best_move = board.san(result["pv"][0]) if "pv" in result else move
                annotated_game.append((best_move, move))
            annotated_games.append(annotated_game)
    finally:
        engine.quit()
    return annotated_games

# Process games with board features
def process_games_with_board_features(games, move_to_id):
    inputs, outputs, board_features = [], [], []
    for game in games:
        board = chess.Board()
        tokenized_game = [move_to_id[move] for move in game]
        for i in range(1, len(tokenized_game)):
            inputs.append(tokenized_game[:i])
            outputs.append(tokenized_game[i])
            board_features.append(board_to_tensor(board))
            board.push_san(game[i - 1])
    return inputs, outputs, board_features

# Fine-tune model
def fine_tune_model(base_model, annotated_inputs, annotated_outputs, annotated_board_features, save_path, epochs=5):
    # Pad inputs
    max_length = max([len(inp) for inp in annotated_inputs])
    annotated_inputs = pad_sequences(annotated_inputs, maxlen=max_length, padding='post')

    # Split into training and validation sets
    X_train_input, X_val_input, y_train_output, y_val_output, X_train_board, X_val_board = train_test_split(
        annotated_inputs, annotated_outputs, annotated_board_features, test_size=0.2, random_state=42
    )

    # Fine-tune the model
    base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path, save_best_only=True
    )

    history = base_model.fit(
        [X_train_input, X_train_board], y_train_output,
        validation_data=([X_val_input, X_val_board], y_val_output),
        batch_size=64,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )

    return history

# Paths and parameters
pgn_file = "../PGN Games/large_lichess_games.pgn"
stockfish_path = "../Stockfish/stockfish/stockfish.exe"
num_games_per_elo = 1000
elo_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]

# Extract and annotate games
balanced_games = []
for elo_range in elo_ranges:
    balanced_games.extend(extract_games_by_elo(pgn_file, elo_range, num_games_per_elo))

# Load existing vocabulary from the base model
with open("models/base_transformer_100k_games/move_to_id.json", "r") as f:
    move_to_id = json.load(f)

with open("models/base_transformer_100k_games/id_to_move.json", "r") as f:
    id_to_move = json.load(f)

# Annotate games with Stockfish
annotated_games = annotate_with_stockfish(balanced_games, stockfish_path, depth=10)

# Process annotated games with the loaded move_to_id
annotated_inputs, annotated_outputs, annotated_board_features = process_games_with_board_features(annotated_games, move_to_id)

# Fine-tune the base model
base_model_path = "models/base_transformer_100k_games/transformer_model_with_board_checkpoint.h5"
base_model = tf.keras.models.load_model(base_model_path)

fine_tuned_model_path = "models/fine_tuned_transformer_checkpoint.h5"
history = fine_tune_model(
    base_model,
    annotated_inputs,
    annotated_outputs,
    annotated_board_features,
    fine_tuned_model_path,
    epochs=5
)

print("Fine-tuning complete.")


