#loss: 0.3692 - accuracy: 0.9224 - val_loss: 0.4100 - val_accuracy: 0.9181 for fine_tuned_elo_0_999.h5 - 15k games used
#loss: 0.4495 - accuracy: 0.9058 - val_loss: 0.4782 - val_accuracy: 0.9036 for fine_tuned_elo_1000_1500.h5 - 15k games used
#loss: 0.5153 - accuracy: 0.8927 - val_loss: 0.5559 - val_accuracy: 0.8888 for fine_tuned_elo_1500_2000.h5 - 15k games used
#loss: 0.5937 - accuracy: 0.8781 - val_loss: 0.6249 - val_accuracy: 0.8761 for fine_tuned_elo_2000_plus.h5 - 15k games used

import chess.pgn
import json
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Load move mappings
with open("models/base_transformer_50k_games_Models/move_to_id.json", "r") as f:
    move_to_id = json.load(f)

with open("models/base_transformer_50k_games_Models/id_to_move.json", "r") as f:
    id_to_move = json.load(f)
id_to_move = {int(k): v for k, v in id_to_move.items()}  # Ensure keys are integers

# Constants
max_length = 485  # Original data padding length

# Function to extract games for a specific ELO range
def extract_games_by_elo_range(pgn_file, min_elo=0, max_elo=float('inf'), limit=15000):
    with open(pgn_file, "r") as file:
        games = []
        game_count = 0
        while game_count < limit:
            game = chess.pgn.read_game(file)
            if game is None:
                break
            
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            if min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo:
                moves = []
                board = game.board()
                for move in game.mainline_moves():
                    moves.append(board.san(move))
                    board.push(move)
                games.append(moves)
                game_count += 1
        print(f"Extracted {len(games)} games for ELO range {min_elo}-{max_elo}")
    return games

# Load and tokenize dataset for a specific ELO range
def prepare_data(pgn_file, min_elo, max_elo, move_to_id, max_length):
    games = extract_games_by_elo_range(pgn_file, min_elo, max_elo)
    tokenized_games = [[move_to_id.get(move, 0) for move in game if move in move_to_id] for game in games]
    # Pad sequences to one more than the model's expected input length
    padded_games = pad_sequences(tokenized_games, maxlen=max_length + 1, padding='post', truncating='post')
    print(f"Padded games shape: {padded_games.shape}")  # Debugging log
    return padded_games


# Prepare training and validation data
def prepare_train_val_data(padded_games, test_size=0.2):
    # Input sequences should have 485 columns (matching the model input)
    X = padded_games[:, :-1]  # All but the last token for input
    y = padded_games[:, 1:]   # All but the first token for target
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Fine-tuning function
def fine_tune_model(base_model_path, pgn_file, min_elo, max_elo, save_path, batch_size=64, epochs=5):
    # Load base model
    model = load_model(base_model_path)
    print(f"Loaded base model with expected input shape: {model.input_shape}")

    # Prepare data
    padded_games = prepare_data(pgn_file, min_elo, max_elo, move_to_id, max_length)
    X_train, X_val, y_train, y_val = prepare_train_val_data(padded_games)

    # Debugging logs
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fine-tune the model
    checkpoint = ModelCheckpoint(save_path, save_best_only=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    return history

# Fine-tune for different ELO ranges
pgn_file = "../PGN Games/partial_lichess_games_50k-450k.pgn"

elo_ranges = [
    (0, 999, "models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models/fine_tuned_elo_0_999.h5"),
    (1000, 1500, "models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models/fine_tuned_elo_1000_1500.h5"),
    (1500, 2000, "models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models/fine_tuned_elo_1500_2000.h5"),
    (2000, float('inf'), "models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models/fine_tuned_elo_2000_plus.h5")
]

for min_elo, max_elo, save_path in elo_ranges:
    print(f"Fine-tuning model for ELO range {min_elo}-{max_elo}")
    fine_tune_model(
        base_model_path="models/base_transformer_50k_games_Models/model_checkpoint.h5",
        pgn_file=pgn_file,
        min_elo=min_elo,
        max_elo=max_elo,
        save_path=save_path,
    )
    print(f"Model fine-tuned and saved to {save_path}")




