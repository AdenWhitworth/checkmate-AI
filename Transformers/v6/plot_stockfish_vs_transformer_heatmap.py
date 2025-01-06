import random
import chess
import chess.pgn
from stockfish import Stockfish
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

# Stockfish skill-to-ELO mapping
stockfish_elo_mapping = {
    1: 800, 2: 900, 3: 1000, 4: 1100, 5: 1200, 6: 1300, 7: 1400, 8: 1500,
    9: 1600, 10: 1700, 11: 1800, 12: 1900, 13: 2000, 14: 2100, 15: 2200,
    16: 2300, 17: 2400, 18: 2500, 19: 2600, 20: 2700
}

# Load move mappings
with open("../Transformers/models/base_transformer_50k_games_Models/move_to_id.json", "r") as f:
    moveToId = json.load(f)
with open("../Transformers/models/base_transformer_50k_games_Models/id_to_move.json", "r") as f:
    idToMove = json.load(f)
idToMove = {int(k): v for k, v in idToMove.items()}  # Ensure keys are integers

# Load the trained model
model = load_model("../Transformers/models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models/fine_tuned_elo_2000_plus.h5")
max_length = model.input_shape[1]

# Initialize Stockfish
stockfish = Stockfish("../Stockfish/stockfish/stockfish.exe")

# Predict next move function
def predict_next_move(model, move_history, move_to_id, id_to_move, max_length, board):
    tokenized_history = [move_to_id.get(move, 0) for move in move_history]
    padded_history = pad_sequences([tokenized_history], maxlen=max_length, padding="post")
    predictions = model.predict(padded_history, verbose=0)
    move_probabilities = predictions[0, len(move_history)]
    sorted_indices = tf.argsort(move_probabilities, direction="DESCENDING").numpy()

    for predicted_move_id in sorted_indices:
        predicted_move = id_to_move.get(predicted_move_id, "<unknown>")
        if predicted_move in [board.san(move) for move in board.legal_moves]:
            return predicted_move

    # Fallback to random move
    return random.choice([board.san(move) for move in board.legal_moves])

# Parameters for the heatmap
num_trials = 1 # Number of games per skill level
skill_levels = range(1, 21)  # Stockfish skill levels (1 to 20)
results_matrix = np.zeros((len(skill_levels), num_trials))  # Rows = skill levels, Cols = trials

# Play games for each skill level
for i, skill_level in enumerate(skill_levels):
    stockfish.set_skill_level(skill_level)
    for trial in range(num_trials):
        board = chess.Board()
        move_history = []

        # Play the game
        while not board.is_game_over():
            if board.turn:  # Model's turn
                predicted_move = predict_next_move(model, move_history, moveToId, idToMove, max_length, board)
                if predicted_move in [board.san(move) for move in board.legal_moves]:
                    board.push_san(predicted_move)
                    move_history.append(predicted_move)
            else:  # Stockfish's turn
                stockfish.set_fen_position(board.fen())
                stockfish_move = stockfish.get_best_move()
                stockfish_move_obj = chess.Move.from_uci(stockfish_move)
                if stockfish_move_obj in board.legal_moves:
                    san_move = board.san(stockfish_move_obj)
                    board.push(stockfish_move_obj)
                    move_history.append(san_move)

        # Record the result
        game_result = board.result()  # "1-0", "0-1", or "1/2-1/2"
        print(board.fen(), game_result)
        if game_result == "1-0":
            results_matrix[i, trial] = 1  # Win
        elif game_result == "1/2-1/2":
            results_matrix[i, trial] = 0.5  # Draw
        else:
            results_matrix[i, trial] = 0  # Loss

# Stockfish ELO levels for y-axis
elo_levels = [stockfish_elo_mapping[i] for i in skill_levels]

# Create the heatmap
plt.figure(figsize=(12, 8))
plt.imshow(results_matrix, cmap="YlGnBu", aspect="auto", interpolation="nearest")

# Add color bar
cbar = plt.colorbar()
cbar.set_label("Performance (1=Win, 0.5=Draw, 0=Loss)")

# Label the axes
plt.title("Model Performance vs Stockfish Skill Level")
plt.xlabel("Trial")
plt.ylabel("Stockfish ELO")
plt.xticks(ticks=np.arange(results_matrix.shape[1]), labels=[f"Trial {i+1}" for i in range(num_trials)])
plt.yticks(ticks=np.arange(results_matrix.shape[0]), labels=elo_levels)

# Annotate cells with values
for i in range(results_matrix.shape[0]):
    for j in range(results_matrix.shape[1]):
        plt.text(j, i, f"{results_matrix[i, j]:.1f}", ha="center", va="center", color="black")

# Show the plot
plt.tight_layout()
plt.show()

