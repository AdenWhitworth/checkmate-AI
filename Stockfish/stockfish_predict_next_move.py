from stockfish import Stockfish

# Set the path to your Stockfish executable
stockfish_path = "../Stockfish/stockfish/stockfish.exe"

# Initialize the Stockfish engine
stockfish = Stockfish(stockfish_path)

# Set the strength level of the engine
stockfish.set_skill_level(20)  # 0 to 20, where 20 is the strongest

# Set up a chess position using FEN
fen_position = "r1bqkbnr/pppppppp/n7/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"
stockfish.set_fen_position(fen_position)

# Get the best move
best_move = stockfish.get_best_move()
print(f"Best move: {best_move}")

# Evaluate the current position
evaluation = stockfish.get_evaluation()
print(f"Position evaluation: {evaluation}")

# Play a move and update the board
stockfish.make_moves_from_current_position(["e2e4", "e7e5", "g1f3"])
print(f"Updated FEN: {stockfish.get_fen_position()}")
