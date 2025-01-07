import onnxruntime as ort
from tensorflow.keras.preprocessing.sequence import pad_sequences
import chess
import random
import json
import numpy as np
import tensorflow as tf

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

def predict_next_move(session, move_history, move_to_id, id_to_move, max_length, board):
    """
    Predict the next move using a trained ONNX model.

    Args:
        session (onnxruntime.InferenceSession): ONNX inference session.
        move_history (list): List of SAN moves representing the move history.
        move_to_id (dict): Mapping of moves to unique IDs.
        id_to_move (dict): Mapping of IDs to moves.
        max_length (int): Maximum length of the input sequence for the model.
        board (chess.Board): Current board state.

    Returns:
        tuple: Predicted SAN move, its probability, and sorted probabilities with associated moves.
    """
    # Tokenize the move history
    tokenized_history = [move_to_id.get(move, 0) for move in move_history]

    # Pad the tokenized history to match the model's input length
    padded_history = pad_sequences([tokenized_history], maxlen=max_length, padding="post")

    # Perform inference
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: np.array(padded_history, dtype=np.float32)})[0]

    # Extract probabilities for the last position in the sequence
    move_probabilities = predictions[0, -1]  # Use the last row of the last batch

    # Sort moves by predicted probability while maintaining associations
    sorted_indices = np.argsort(move_probabilities)[::-1]  # Indices of moves sorted by probability (descending)
    sorted_probabilities = [move_probabilities[idx] for idx in sorted_indices]  # Sorted probabilities
    sorted_moves = [id_to_move.get(int(idx), "<unknown>") for idx in sorted_indices]  # Sorted moves by index

    # Find the first legal move from the sorted moves
    for move_san, prob in zip(sorted_moves, sorted_probabilities):
        try:
            move_uci = board.parse_san(move_san).uci()
            if move_uci in [m.uci() for m in board.legal_moves]:
                return move_uci, prob, list(zip(sorted_moves, sorted_probabilities))  # Return UCI move
        except ValueError:
            continue  # Skip invalid SAN moves

    # Fallback if no legal moves are found
    return None, 0, list(zip(sorted_moves, sorted_probabilities))

def setup_and_predict_move(model_path, move_to_id_path, id_to_move_path, move_history):
    """
    Set up the ONNX model, mappings, and board, and predict the next move.

    Args:
        model_path (str): Path to the ONNX model file.
        move_to_id_path (str): Path to the JSON file mapping moves to IDs.
        id_to_move_path (str): Path to the JSON file mapping IDs to moves.
        move_history (list): List of SAN moves representing the move history.

    Returns:
        tuple: The predicted move, updated board, and sorted probabilities with associated moves.
    """
    # Load mappings
    move_to_id, id_to_move = load_mappings(move_to_id_path, id_to_move_path)

    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Determine max length from the model's input shape
    max_length = session.get_inputs()[0].shape[1]

    # Initialize the board and apply the move history
    board = chess.Board()
    for move in move_history:
        board.push_san(move)

    # Predict the next move
    predicted_move, predicted_move_prob, sorted_probabilities = predict_next_move(
        session, move_history, move_to_id, id_to_move, max_length, board
    )

    # Apply the move to the board
    if predicted_move:
        move_obj = chess.Move.from_uci(predicted_move)
        board.push(move_obj)
    else:
        # Fallback to a random legal move
        print("Fallback to random move for no legal moves.")
        fallback_move = random.choice(list(board.legal_moves))
        board.push(fallback_move)
        predicted_move = fallback_move.uci()

    return predicted_move, predicted_move_prob, board, sorted_probabilities

if __name__ == "__main__":
    # Paths to ONNX model and mappings
    model_path = "../models/base_transformer_50k_games_Models/onnx_models/model_elo_2000_plus.onnx"
    move_to_id_path = "../models/base_transformer_50k_games_Models/move_to_id.json"
    id_to_move_path = "../models/base_transformer_50k_games_Models/id_to_move.json"

    # Example move history
    move_history = ["d4", "c5", "d5", "Nf6"]

    # Predict the next move
    predicted_move, predicted_move_prob, updated_board, sorted_probabilities = setup_and_predict_move(
        model_path, move_to_id_path, id_to_move_path, move_history
    )

    print(f"Predicted move: {predicted_move} : {predicted_move_prob:.6f}")
    print("Top 10 Sorted probabilities with moves:")
    for move, prob in sorted_probabilities[:10]:  # Slicing the list to get top 10 entries
        print(f"{move}: {prob:.6f}")
    print(f"Updated board after move:\n{updated_board}")