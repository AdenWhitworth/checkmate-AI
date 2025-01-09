"""
Chess Position Evaluation and Analysis Script
=============================================

This script leverages the Stockfish chess engine to evaluate chess positions, retrieve top moves, 
and analyze specific moves from a given board state. It is designed for chess enthusiasts, 
developers, and AI researchers who need an efficient way to evaluate and understand chess positions.

Features
--------
1. **Stockfish Initialization**:
   - Configures the Stockfish engine with customizable threads and skill levels for efficient evaluation.

2. **Position Evaluation**:
   - Evaluates a given chess position in FEN (Forsyth-Edwards Notation) format.
   - Retrieves the overall evaluation of the position and evaluations of the top legal moves.

3. **Move Analysis**:
   - Evaluates specific moves from a given FEN position to determine their quality.

4. **Result Display**:
   - Formats and prints the evaluation results in a human-readable format, including centipawn (CP) and mate scores.

5. **Modular Design**:
   - Each functionality is encapsulated in reusable functions to enhance readability, reusability, and maintainability.

6. **Customizable Depth and Moves**:
   - Users can specify the depth of Stockfish’s search and the number of top moves to retrieve.

Inputs
------
- **FEN String**: A string representing the chess board state.
- **Move**: UCI (Universal Chess Interface) notation of a move to analyze (optional).
- **Stockfish Configuration**: Path to the Stockfish executable, number of threads, and skill level.

Outputs
-------
- **Overall Position Evaluation**: The assessment of the given position in terms of centipawn or mate score.
- **Top Moves Evaluation**: List of the top legal moves and their evaluations.
- **Specific Move Evaluation**: Evaluation of a specific move from the given position.
"""

from stockfish import Stockfish

def initialize_stockfish(stockfish_path, threads=8, skill_level=10):
    """
    Initialize the Stockfish engine with given parameters.

    Args:
        stockfish_path (str): Path to the Stockfish executable.
        threads (int): Number of threads to use.
        skill_level (int): Skill level of Stockfish (1-20).

    Returns:
        Stockfish: Initialized Stockfish engine.
    """
    return Stockfish(stockfish_path, parameters={"Threads": threads, "Skill Level": skill_level})


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
        return f"{evaluation['value'] / 100:.2f} pawns"
    elif evaluation["type"] == "mate":
        return f"Mate in {evaluation['value']}"
    return "Unknown evaluation"


def get_top_moves(stockfish, num_top_moves):
    """
    Get top moves from Stockfish along with their evaluations.

    Args:
        stockfish (Stockfish): Initialized Stockfish engine.
        num_top_moves (int): Number of top moves to retrieve.

    Returns:
        list[dict]: List of dictionaries with move and evaluation data.
    """
    top_moves = stockfish.get_top_moves(num_top_moves)
    return [
        {
            "move": move["Move"],
            "evaluation": format_evaluation(
                {"type": "cp", "value": move["Centipawn"]}
                if "Centipawn" in move
                else {"type": "mate", "value": move["Mate"]}
            ),
        }
        for move in top_moves
    ]


def evaluate_fen(stockfish, fen, depth=10, num_top_moves=5):
    """
    Evaluate a FEN position using Stockfish.

    Args:
        stockfish (Stockfish): Initialized Stockfish engine.
        fen (str): The FEN string to evaluate.
        depth (int): Depth to search for evaluation.
        num_top_moves (int): Number of top moves to evaluate.

    Returns:
        dict: A dictionary containing the overall evaluation and top moves.
    """
    stockfish.set_fen_position(fen)
    stockfish.set_depth(depth)

    overall_eval = format_evaluation(stockfish.get_evaluation())
    top_moves = get_top_moves(stockfish, num_top_moves)

    return {"overall_evaluation": overall_eval, "top_moves": top_moves}


def evaluate_move(stockfish, fen, move, depth=10):
    """
    Evaluate a specific move from a given FEN position using Stockfish.

    Args:
        stockfish (Stockfish): Initialized Stockfish engine.
        fen (str): The FEN string to evaluate from.
        move (str): UCI notation of the move to evaluate.
        depth (int): Depth to search for evaluation.

    Returns:
        str: Formatted evaluation of the move.
    """
    stockfish.set_fen_position(fen)
    stockfish.make_moves_from_current_position([move])
    evaluation = stockfish.get_evaluation()
    return format_evaluation(evaluation)


def display_results(results):
    """
    Display the evaluation results in a readable format.

    Args:
        results (dict): Evaluation results containing overall evaluation and top moves.

    Returns:
        None
    """
    print(f"Overall Evaluation: {results['overall_evaluation']}")
    print("Top Moves and Evaluations:")
    for move in results["top_moves"]:
        print(f"Move: {move['move']}, Evaluation: {move['evaluation']}")


if __name__ == "__main__":
    # FEN position to evaluate
    fen = "r2q1rk1/pp1nbppp/3p1n2/2p1p3/2P1P3/2N1BN2/PPQ2PPP/R3K2R w KQ - 0 10"

    # Path to Stockfish executable
    stockfish_path = "../../../Stockfish/stockfish/stockfish.exe"

    # Initialize Stockfish engine
    stockfish = initialize_stockfish(stockfish_path)

    # Evaluate the FEN position
    results = evaluate_fen(stockfish, fen, depth=10, num_top_moves=5)

    # Display the evaluation results
    display_results(results)

    # Evaluate a specific move
    specific_move = "a2a3"
    move_evaluation = evaluate_move(stockfish, fen, specific_move)
    print(f"\nEvaluation for move {specific_move}: {move_evaluation}")
