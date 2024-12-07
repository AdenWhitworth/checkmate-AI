import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import chess

# Load move mappings
with open("../Transformers/models/base_transformer_50k_games_Models/move_to_id.json", "r") as f:
    moveToId = json.load(f)

with open("../Transformers/models/base_transformer_50k_games_Models/id_to_move.json", "r") as f:
    idToMove = json.load(f)

# Ensure JSON keys are integers
idToMove = {int(k): v for k, v in idToMove.items()}

# Load the trained model
model = load_model("../Transformers/models/base_transformer_50k_games_Models/model_checkpoint.h5")
max_length = model.input_shape[1]

# Prediction function
def predict_move_probabilities(model, move_history, move_to_id, max_length):
    # Tokenize the move history
    tokenized_history = [move_to_id.get(move, 0) for move in move_history]
    # Pad the tokenized history
    padded_history = pad_sequences([tokenized_history], maxlen=max_length, padding='post')
    # Predict probabilities
    predictions = model.predict(padded_history, verbose=0)
    move_probabilities = predictions[0, len(move_history)]  # Get probabilities for the next move
    return move_probabilities

# Plotting function for top 10 predictions
def plot_top_predictions(predictions, vocab, top_n=10):
    """
    Plot the top N predicted moves and their probabilities.
    """
    # Get the top N moves and probabilities
    sorted_indices = np.argsort(predictions)[-top_n:][::-1]  # Indices of top N predictions (sorted descending)
    top_moves = [vocab.get(idx, "<unknown>") for idx in sorted_indices]  # Map indices to moves
    top_probabilities = predictions[sorted_indices]  # Get probabilities of top moves

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), top_probabilities, alpha=0.7, color='blue', label='Predicted Probabilities')
    plt.xticks(range(top_n), top_moves, rotation=45, ha='right')  # Add move notations as x-axis labels
    plt.title(f"Top {top_n} Predicted Moves")
    plt.xlabel("Moves (Chess Notation)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
#move_history = ['d4', 'c5', 'd5', 'Nf6']  # Example move history
move_history = ['e4', 'e5', 'Nf3', 'Nc6']
actual_move = 'c4'  # Actual next move for reference

# Predict probabilities
predictions = predict_move_probabilities(model, move_history, moveToId, max_length)

# Plot top 10 predictions
plot_top_predictions(predictions, idToMove, top_n=10)






"""import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import chess.pgn
import json
import io

# Load move mappings from JSON files
with open("../Transformers/models/base_transformer_50k_games_Models/move_to_id.json", "r") as f:
    moveToId = json.load(f)

with open("../Transformers/models/base_transformer_50k_games_Models/id_to_move.json", "r") as f:
    idToMove = json.load(f)

# Ensure JSON keys are converted to the correct types
idToMove = {int(k): v for k, v in idToMove.items()}  # Convert keys to int

# Load the trained model
model = load_model("../Transformers/models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models/fine_tuned_elo_1500_2000.h5")
max_length = model.input_shape[1]

def predict_move_probabilities(model, move_history, move_to_id, max_length):
    # Tokenize the move history
    tokenized_history = [move_to_id.get(move, 0) for move in move_history]
    
    # Pad the tokenized history to match the model's input length
    padded_history = pad_sequences([tokenized_history], maxlen=max_length, padding='post')
    
    # Predict the next move probabilities
    predictions = model.predict(padded_history, verbose=0)  # Disable verbose output
    move_probabilities = predictions[0, len(move_history)]
    
    return move_probabilities

def plot_move_probabilities(predictions, actual_move_id, vocab, top_n=20):
    # Get the top N moves and their probabilities
    sorted_indices = np.argsort(predictions)[-top_n:][::-1]  # Indices of top N predictions
    top_moves = [vocab.get(idx, "<unknown>") for idx in sorted_indices]  # Get move notations
    top_probabilities = predictions[sorted_indices]  # Probabilities of the top moves
    
    # Plot the top N moves
    plt.figure(figsize=(15, 5))
    plt.bar(range(top_n), top_probabilities, alpha=0.7, color='blue', label='Predicted Probabilities')
    
    # Highlight the actual move if it's in the top N
    if actual_move_id in sorted_indices:
        actual_index = np.where(sorted_indices == actual_move_id)[0][0]
        plt.bar(actual_index, predictions[actual_move_id], alpha=0.9, color='red', label='Actual Move')
    
    # Add move notations as X-axis labels
    plt.xticks(range(top_n), top_moves, rotation=45, ha='right')
    
    # Add title and labels
    actual_move_notation = vocab.get(actual_move_id, "<unknown>")
    plt.title(f"Predicted Probabilities for Moves (Actual Move: {actual_move_notation})")
    plt.xlabel("Moves (Chess Notation)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

move_history = ['d4', 'c5', 'd5', 'Nf6']
actual_move = 'c4'

# Predict probabilities
predictions = predict_move_probabilities(model, move_history, moveToId, max_length)

# Get the ID of the actual move
actual_move_id = moveToId.get(actual_move, 0)

# Plot predicted probabilities
plot_move_probabilities(predictions, actual_move_id, idToMove, top_n=20)"""


