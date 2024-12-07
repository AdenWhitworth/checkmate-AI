from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import chess.pgn
import chess
import random
import json

# Load move mappings from JSON files
with open("../models/base_transformer_50k_games_Models/move_to_id.json", "r") as f:
    moveToId = json.load(f)

with open("../models/base_transformer_50k_games_Models/id_to_move.json", "r") as f:
    idToMove = json.load(f)

# Ensure JSON keys are converted to the correct types
idToMove = {int(k): v for k, v in idToMove.items()}  # Convert keys to int

# Load the trained model
model = load_model("../models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models/fine_tuned_elo_1500_2000.h5")
max_length = model.input_shape[1]

def predict_next_move(model, move_history, move_to_id, id_to_move, max_length, board):
    # Tokenize the move history
    tokenized_history = [move_to_id.get(move, 0) for move in move_history]
    
    # Pad the tokenized history to match the model's input length
    padded_history = pad_sequences([tokenized_history], maxlen=max_length, padding='post')
    
    # Predict the next move probabilities
    predictions = model.predict(padded_history, verbose=0)  # Disable verbose output
    move_probabilities = predictions[0, len(move_history)]
    
    # Sort moves by predicted probability
    sorted_indices = tf.argsort(move_probabilities, direction="DESCENDING").numpy()

    print(sorted_indices)
    
    # Iterate through predicted moves to find the first legal one
    for predicted_move_id in sorted_indices:
        predicted_move = id_to_move.get(predicted_move_id, "<unknown>")
        if predicted_move in [board.san(move) for move in board.legal_moves]:
            return predicted_move

    # Fallback: return a random legal move if no predicted moves are legal
    return random.choice([board.san(move) for move in board.legal_moves])

# Example move history
#move_history = ['e4', 'e5', 'Nf3', 'Nc6']
move_history = ['d4', 'c5', 'd5', 'Nf6']

# Initialize the board
board = chess.Board()
for move in move_history:
    board.push_san(move)

# Predict the next move
predicted_move = predict_next_move(model, move_history, moveToId, idToMove, max_length, board)

# Apply the move and update the board
if predicted_move in [board.san(move) for move in board.legal_moves]:
    print(f"Predicted move {predicted_move} is legal.")
    board.push_san(predicted_move)
else:
    print(f"Predicted move {predicted_move} is illegal. Choosing a random legal move.")
    random_legal_move = random.choice([board.san(move) for move in board.legal_moves])
    board.push_san(random_legal_move)
    predicted_move = random_legal_move

print(f"Updated board after move:\n{board}")




