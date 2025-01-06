"""
Transformer-based Chess Move Prediction Model.

This script:
1. Filters chess games by ELO range from a PGN file.
2. Builds a vocabulary of chess moves and tokenizes games.
3. Prepares training and validation datasets for a Transformer model.
4. Defines, trains, and evaluates a Transformer-based model for predicting the next chess move.

Results:
- 15k Games: loss: 0.8023 - accuracy: 0.8491 - val_loss: 0.8449 - val_accuracy: 0.8469
- 50k Games: 0.5244 - accuracy: 0.8929 - val_loss: 0.5491 - val_accuracy: 0.8910
"""

import chess.pgn
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, Dropout, LayerNormalization
import json

def extract_games_by_elo_range(pgn_file, min_elo=0, max_elo=float('inf')):
    """
    Extract games from a PGN file within a specific ELO range.

    Args:
        pgn_file (str): Path to the PGN file.
        min_elo (int): Minimum ELO rating for filtering games.
        max_elo (int): Maximum ELO rating for filtering games.

    Returns:
        list: List of games where each game is a list of SAN moves.
    """
    games = []
    with open(pgn_file, "r") as file:
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break

            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            
            if min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo:
                board = game.board()
                moves = [board.san(move) for move in game.mainline_moves()]
                games.append(moves)

    return games

def build_vocab(games):
    """
    Build a vocabulary from a list of games.

    Args:
        games (list): List of games where each game is a list of SAN moves.

    Returns:
        tuple: (move_to_id, id_to_move) mappings.
    """
    vocab = defaultdict(int)
    for game in games:
        for move in game:
            vocab[move] += 1

    move_to_id = {move: idx for idx, move in enumerate(vocab.keys())}
    id_to_move = {idx: move for move, idx in move_to_id.items()}
    return move_to_id, id_to_move

def tokenize_games(games, move_to_id):
    """
    Tokenize games using a move-to-ID mapping.

    Args:
        games (list): List of games where each game is a list of SAN moves.
        move_to_id (dict): Mapping of moves to unique IDs.

    Returns:
        list: List of tokenized games.
    """
    return [[move_to_id[move] for move in game] for game in games]


def build_transformer_model(vocab_size, max_length, embed_dim=128, num_heads=4, ff_dim=128):
    """
    Build a Transformer model for next-move prediction.

    Args:
        vocab_size (int): Size of the vocabulary.
        max_length (int): Maximum length of input sequences.
        embed_dim (int): Dimensionality of embedding layer.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of feed-forward layers.

    Returns:
        tf.keras.Model: Compiled Transformer model.
    """
    inputs = Input(shape=(max_length,))
    x = Embedding(vocab_size, embed_dim)(inputs)
    x = LayerNormalization()(x)

    for _ in range(3):  # Number of Transformer layers
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        attention_output = Dropout(0.1)(attention_output)
        x = LayerNormalization()(x + attention_output)

        ff_output = Dense(ff_dim, activation='relu')(x)
        ff_output = Dropout(0.1)(ff_output)
        x = LayerNormalization()(x + ff_output)

    outputs = Dense(vocab_size, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

def main(pgn_file, model_dir):
    """
    Main function to preprocess data, train, and evaluate the Transformer model.

    Args:
        pgn_file (str): Path to the PGN file containing the dataset.
        model_file (str): Path to save the trained model.
    """
    
    # Extract games by ELO ranges
    games_below_1000 = extract_games_by_elo_range(pgn_file, max_elo=999)
    games_1000_to_1500 = extract_games_by_elo_range(pgn_file, min_elo=1000, max_elo=1500)
    games_1500_to_2000 = extract_games_by_elo_range(pgn_file, min_elo=1500, max_elo=2000)
    games_above_2000 = extract_games_by_elo_range(pgn_file, min_elo=2000)

    # Combine all games and build vocabulary
    all_games = games_below_1000 + games_1000_to_1500 + games_1500_to_2000 + games_above_2000
    move_to_id, id_to_move = build_vocab(all_games)

    # Save mappings
    with open(f"{model_dir}/move_to_id.json", "w") as f:
        json.dump(move_to_id, f)
    with open(f"{model_dir}/id_to_move.json", "w") as f:
        json.dump(id_to_move, f)

    # Tokenize games and pad sequences
    tokenized_games = tokenize_games(all_games, move_to_id)
    max_length = max(len(game) for game in tokenized_games)
    padded_games = pad_sequences(tokenized_games, maxlen=max_length, padding='post')

    # Split data into training and validation sets
    X_train, X_val = train_test_split(padded_games, test_size=0.2, random_state=42)
    X_train_input, y_train_output = X_train[:, :-1], X_train[:, 1:]
    X_val_input, y_val_output = X_val[:, :-1], X_val[:, 1:]

    # Build and train the model
    vocab_size = len(move_to_id)
    adjusted_max_length = max_length - 1
    model = build_transformer_model(vocab_size, adjusted_max_length)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Model training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{model_dir}/model_checkpoint.h5",
        save_best_only=True
    )
    model.fit(
        X_train_input, y_train_output,
        validation_data=(X_val_input, y_val_output),
        batch_size=64,
        epochs=10,
        callbacks=[checkpoint_callback]
    )

    # Evaluate the model
    model.evaluate(X_val_input, y_val_output)

    # Save the final model
    model.save(f"{model_dir}/final_model.h5")

if __name__ == "__main__":
    # File paths
    pgn_file = "../../../PGN Games/pgns/partial_lichess_games_50k.pgn"
    model_dir = "models/base_transformer_50k_games_Models"
    main(pgn_file, model_dir)