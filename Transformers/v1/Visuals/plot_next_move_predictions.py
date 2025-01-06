import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os

# Add the parent directory to PYTHONPATH
parent_dir = os.path.abspath("..")  # Adjust if your directory structure differs
sys.path.append(parent_dir)

# Now you can import the module
from testing.predict_next_move import setup_and_predict_move



def plot_top_predictions(sorted_probabilities, top_n=10):
    """
    Plot the top N predicted moves and their probabilities.

    Args:
        sorted_probabilities (list): List of tuples containing moves and their probabilities.
        top_n (int): Number of top predictions to plot.
    """
    # Extract top moves and probabilities
    top_moves = [move for move, prob in sorted_probabilities[:top_n]]
    top_probabilities = [prob for move, prob in sorted_probabilities[:top_n]]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), top_probabilities, alpha=0.7, color='blue')
    plt.xticks(range(top_n), top_moves, rotation=45, ha='right')
    plt.title(f"Top {top_n} Predicted Moves")
    plt.xlabel("Moves")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Paths to model and mapping files
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

    plot_top_predictions(sorted_probabilities, top_n=10)
