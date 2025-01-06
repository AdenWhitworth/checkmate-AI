"""
Script to predict the next move in a chess game using a trained TensorFlow model.

This script:
1. Loads a pre-trained TensorFlow model for chess move prediction.
2. Preprocesses a given move history into a format suitable for model input.
3. Predicts the next move and ensures it is legal according to the current board state.

Functions:
- load_mappings: Loads move mappings from JSON files.
- predict_next_move: Predicts the next move given a move history and board state.
- setup_and_predict_move: Orchestrates the loading, initialization, and prediction process.

Usage:
- Update the paths to the model and JSON files.
- Provide a move history to test predictions.
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import chess
import random
import json

def load_mappings(move_to_id_path, id_to_move_path):
    """
    Load move mappings from JSON files and ensure correct key-value types.

    Args:
        move_to_id_path (str): Path to the JSON file mapping moves to IDs.
        id_to_move_path (str): Path to the JSON file mapping IDs to moves.

    Returns:
        tuple: A tuple containing the `move_to_id` and `id_to_move` mappings.
    """
    with open(move_to_id_path, "r") as f:
        move_to_id = json.load(f)

    with open(id_to_move_path, "r") as f:
        id_to_move = json.load(f)

    # Ensure keys in `id_to_move` are integers
    id_to_move = {int(k): v for k, v in id_to_move.items()}
    return move_to_id, id_to_move


def predict_next_move(model, move_history, move_to_id, id_to_move, max_length, board):
    """
    Predict the next move using a trained model.

    Args:
        model (tf.keras.Model): Pre-trained TensorFlow model for move prediction.
        move_history (list): List of SAN moves representing the move history.
        move_to_id (dict): Mapping of moves to unique IDs.
        id_to_move (dict): Mapping of IDs to moves.
        max_length (int): Maximum length of the input sequence for the model.
        board (chess.Board): Current board state.

    Returns:
        str: Predicted SAN move.
    """
    # Tokenize the move history
    tokenized_history = [move_to_id.get(move, 0) for move in move_history]

    # Pad the tokenized history to match the model's input length
    padded_history = pad_sequences([tokenized_history], maxlen=max_length, padding="post")

    # Predict the next move probabilities
    predictions = model.predict(padded_history, verbose=0)
    move_probabilities = predictions[0]

    # Sort moves by predicted probability
    sorted_indices = tf.argsort(move_probabilities, direction="DESCENDING").numpy()

    # Find the first legal move from the predictions
    for predicted_move_id in sorted_indices:
        predicted_move = id_to_move.get(predicted_move_id, "<unknown>")
        if predicted_move in [board.san(move) for move in board.legal_moves]:
            return predicted_move

    # Fallback: return a random legal move if no predicted moves are legal
    return random.choice([board.san(move) for move in board.legal_moves])

def setup_and_predict_move(model_path, move_to_id_path, id_to_move_path, move_history):
    """
    Set up the model, mappings, and board, and predict the next move.

    Args:
        model_path (str): Path to the pre-trained TensorFlow model.
        move_to_id_path (str): Path to the JSON file mapping moves to IDs.
        id_to_move_path (str): Path to the JSON file mapping IDs to moves.
        move_history (list): List of SAN moves representing the move history.

    Returns:
        tuple: The predicted move and the updated board state.
    """
    # Load mappings
    move_to_id, id_to_move = load_mappings(move_to_id_path, id_to_move_path)

    # Load the model with the correct optimizer
    model = load_model(model_path, custom_objects={"Adam": Adam})

    max_length = model.input_shape[1]

    # Initialize the board and apply the move history
    board = chess.Board()
    for move in move_history:
        board.push_san(move)

    # Predict the next move
    predicted_move = predict_next_move(model, move_history, move_to_id, id_to_move, max_length, board)

    # Apply the move to the board
    if predicted_move in [board.san(move) for move in board.legal_moves]:
        board.push_san(predicted_move)
    else:
        # Fallback to a random legal move
        predicted_move = random.choice([board.san(move) for move in board.legal_moves])
        board.push_san(predicted_move)

    return predicted_move, board

if __name__ == "__main__":
    # Paths to model and mappings
    model_path = "../models/base_transformer_50k_games_Models/tune_transformer_15k_games_Models/fine_tuned_elo_2000_plus.h5"
    move_to_id_path = "../models/base_transformer_50k_games_Models/move_to_id.json"
    id_to_move_path = "../models/base_transformer_50k_games_Models/id_to_move.json"

    # Example move history
    move_history = ["d4", "c5", "d5", "Nf6"]

    # Predict the next move
    predicted_move, updated_board = setup_and_predict_move(
        model_path, move_to_id_path, id_to_move_path, move_history
    )

    # Display the result
    print(f"Predicted move: {predicted_move}")
    print(f"Updated board after move:\n{updated_board}")