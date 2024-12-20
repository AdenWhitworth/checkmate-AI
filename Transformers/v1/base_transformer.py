#V1 loss: 0.8023 - accuracy: 0.8491 - val_loss: 0.8449 - val_accuracy: 0.8469 - 15k Games used
#V2 loss: 0.5244 - accuracy: 0.8929 - val_loss: 0.5491 - val_accuracy: 0.8910 - 50k Games used

import chess.pgn
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, Dropout, LayerNormalization
import json

# Function to filter games by ELO range
def extract_games_by_elo_range(pgn_file, min_elo=0, max_elo=float('inf')):
    """Extract games within a specific ELO range."""
    with open(pgn_file, "r") as file:
        games = []
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break
            
            # Extract player ratings
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            
            if min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo:
                moves = []
                board = game.board()
                for move in game.mainline_moves():
                    moves.append(board.san(move))  # Get move in algebraic notation
                    board.push(move)
                games.append(moves)
    return games

# Example usage for different ELO ranges
games_below_1000 = extract_games_by_elo_range("../PGN Games/partial_lichess_games_50k.pgn", max_elo=999)
games_1000_to_1500 = extract_games_by_elo_range("../PGN Games/partial_lichess_games_50k.pgn", min_elo=1000, max_elo=1500)
games_1500_to_2000 = extract_games_by_elo_range("../PGN Games/partial_lichess_games_50k.pgn", min_elo=1500, max_elo=2000)
games_above_2000 = extract_games_by_elo_range("../PGN Games/partial_lichess_games_50k.pgn", min_elo=2000)

# Build vocabulary
def build_vocab(games):
    vocab = defaultdict(int)
    for game in games:
        for move in game:
            vocab[move] += 1
    move_to_id = {move: idx for idx, move in enumerate(vocab.keys())}
    id_to_move = {idx: move for move, idx in move_to_id.items()}
    return move_to_id, id_to_move

# Tokenize games
def tokenize_games(games, move_to_id):
    return [[move_to_id[move] for move in game] for game in games]

# Process all games
all_games = games_below_1000 + games_1000_to_1500 + games_1500_to_2000 + games_above_2000
move_to_id, id_to_move = build_vocab(all_games)
tokenized_games = tokenize_games(all_games, move_to_id)

with open("models/base_transformer_50k_games_Models/move_to_id.json", "w") as f:
    json.dump(move_to_id, f)

with open("models/base_transformer_50k_games_Models/id_to_move.json", "w") as f:
    json.dump(id_to_move, f)

# Pad sequences
max_length = max(len(game) for game in tokenized_games)
padded_games = pad_sequences(tokenized_games, maxlen=max_length, padding='post')

# Split for general bot
X_train, X_val = train_test_split(padded_games, test_size=0.2, random_state=42)

# Prepare inputs and targets
X_train_input = X_train[:, :-1]
y_train_output = X_train[:, 1:]
X_val_input = X_val[:, :-1]
y_val_output = X_val[:, 1:]

def build_transformer_model(vocab_size, max_length, embed_dim=128, num_heads=4, ff_dim=128):  # Set ff_dim to embed_dim
    inputs = Input(shape=(max_length,))
    x = Embedding(vocab_size, embed_dim)(inputs)
    x = LayerNormalization()(x)

    # Add Transformer Blocks
    for _ in range(3):  # Number of transformer layers
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        attention_output = Dropout(0.1)(attention_output)
        x = LayerNormalization()(x + attention_output)
        
        ff_output = Dense(ff_dim, activation='relu')(x)  # Match ff_dim to embed_dim
        ff_output = Dropout(0.1)(ff_output)
        x = LayerNormalization()(x + ff_output)

    outputs = Dense(vocab_size, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# Prepare Inputs and Targets
X_train_input = X_train[:, :-1]  # All but last move as input
y_train_output = X_train[:, 1:]  # All but first move as target
X_val_input = X_val[:, :-1]
y_val_output = X_val[:, 1:]

# Build and compile the general model
adjusted_max_length = max_length - 1
vocab_size = len(move_to_id)
model = build_transformer_model(vocab_size, adjusted_max_length)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/base_transformer_50k_games_Models/model_checkpoint.h5",
    save_best_only=True
)

history = model.fit(
    X_train_input, y_train_output,
    validation_data=(X_val_input, y_val_output),
    batch_size=64,
    epochs=10,
    callbacks=[checkpoint_callback]
)

# Evaluate Model
model.evaluate(X_val_input, y_val_output)

