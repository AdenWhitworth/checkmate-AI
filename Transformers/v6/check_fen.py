import chess
from stockfish import Stockfish

# Path to Stockfish executable
stockfish_path = "../../Stockfish/stockfish/stockfish.exe"

# Initialize Stockfish
stockfish = Stockfish(stockfish_path, parameters={"Threads": 8, "Skill Level": 10})

def evaluate_fen(fen, depth=10, num_top_moves=5):
    """
    Evaluate a FEN position using Stockfish.
    
    Args:
        fen (str): The FEN string to evaluate.
        depth (int): Depth to search for evaluation.
        num_top_moves (int): Number of top moves to evaluate.
        
    Returns:
        dict: A dictionary containing the overall evaluation and top moves.
    """
    stockfish.set_fen_position(fen)
    stockfish.set_depth(depth)

    # Overall evaluation of the position
    evaluation = stockfish.get_evaluation()

    # Evaluation of top legal moves
    top_moves = stockfish.get_top_moves(num_top_moves)

    # Format evaluations
    overall_eval = format_evaluation(evaluation)
    top_moves_eval = [
        {
            "move": move["Move"],
            "evaluation": format_evaluation({"type": "cp", "value": move["Centipawn"]}) if "Centipawn" in move else format_evaluation({"type": "mate", "value": move["Mate"]})
        }
        for move in top_moves
    ]

    return {"overall_evaluation": overall_eval, "top_moves": top_moves_eval}

def format_evaluation(evaluation):
    """
    Format Stockfish evaluation into a readable format.
    
    Args:
        evaluation (dict): Stockfish evaluation dictionary.
        
    Returns:
        str: Formatted evaluation as a string.
    """
    if evaluation["type"] == "cp":
        # Convert centipawns to pawns
        return f"{evaluation['value'] / 100:.2f} (pawns)"
    elif evaluation["type"] == "mate":
        return f"Mate in {evaluation['value']}"
    return "Unknown evaluation"

# Example FEN
fen = "r2q1rk1/pp1nbppp/3p1n2/2p1p3/2P1P3/2N1BN2/PPQ2PPP/R3K2R w KQ - 0 10"

# Evaluate the FEN
result = evaluate_fen(fen, depth=10, num_top_moves=5)

# Print results
print(f"Overall Evaluation: {result['overall_evaluation']}")
print("Top Moves and Evaluations:")
for move in result["top_moves"]:
    print(f"Move: {move['move']}, Evaluation: {move['evaluation']}")

stockfish.set_fen_position(fen)
stockfish.make_moves_from_current_position(["f3d4"])
eval_f3d4 = stockfish.get_evaluation()
format_eval = format_evaluation(eval_f3d4)
print("eval_f3d4", format_eval)
