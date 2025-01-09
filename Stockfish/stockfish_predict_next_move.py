"""
Stockfish Chess Engine Interaction Script

This script demonstrates how to interact with the Stockfish chess engine using the Python `stockfish` library.
It includes functionalities for initializing the engine, setting up a position, evaluating the board, and retrieving
the best move suggested by the engine.

Usage:
- Modify the `stockfish_path` to point to your Stockfish executable.
- Use the `set_up_stockfish` function to initialize the engine.
- Use `get_best_move` to retrieve the best move in the current position.
- Use `evaluate_position` to get the evaluation of the board state.
- Use `make_moves` to play moves and update the board state.

Dependencies:
- The `stockfish` Python library (`pip install stockfish`).
- Stockfish engine binary.

"""

from stockfish import Stockfish

def set_up_stockfish(stockfish_path, skill_level=20):
    """
    Initialize the Stockfish engine and set the skill level.

    Args:
        stockfish_path (str): Path to the Stockfish executable.
        skill_level (int): Engine's skill level (0 to 20, where 20 is the strongest).

    Returns:
        Stockfish: Initialized Stockfish engine instance.
    """
    stockfish = Stockfish(stockfish_path)
    stockfish.set_skill_level(skill_level)
    return stockfish

def set_position(stockfish, fen):
    """
    Set up a chess position on the Stockfish engine using FEN.

    Args:
        stockfish (Stockfish): The Stockfish engine instance.
        fen (str): The FEN string representing the chess position.
    """
    stockfish.set_fen_position(fen)

def get_best_move(stockfish):
    """
    Retrieve the best move from the current position.

    Args:
        stockfish (Stockfish): The Stockfish engine instance.

    Returns:
        str: The best move in UCI format.
    """
    return stockfish.get_best_move()

def evaluate_position(stockfish):
    """
    Evaluate the current board position using Stockfish.

    Args:
        stockfish (Stockfish): The Stockfish engine instance.

    Returns:
        dict: The evaluation of the position (e.g., centipawn score or mate score).
    """
    return stockfish.get_evaluation()

def make_moves(stockfish, moves):
    """
    Make a series of moves from the current position and update the board.

    Args:
        stockfish (Stockfish): The Stockfish engine instance.
        moves (list): List of moves in UCI format.

    Returns:
        str: The updated FEN string after the moves are applied.
    """
    stockfish.make_moves_from_current_position(moves)
    return stockfish.get_fen_position()

def main(stockfish_path):
    """
    Main function to demonstrate Stockfish functionalities.
    
    Args:
        stockfish_path (string): The path to the stockfish instance.
        moves (list): List of moves in UCI format.
    """
    # Initialize Stockfish
    stockfish = set_up_stockfish(stockfish_path, skill_level=20)

    # Set up a position using FEN
    fen_position = "r1bqkbnr/pppppppp/n7/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"
    set_position(stockfish, fen_position)

    # Get the best move
    best_move = get_best_move(stockfish)
    print(f"Best move: {best_move}")

    # Evaluate the current position
    evaluation = evaluate_position(stockfish)
    print(f"Position evaluation: {evaluation}")

    # Make moves and update the board
    updated_fen = make_moves(stockfish, ["e2e4", "e7e5", "g1f3"])
    print(f"Updated FEN: {updated_fen}")

if __name__ == "__main__":
    stockfish_path = "../Stockfish/stockfish/stockfish.exe"
    main(stockfish_path)
