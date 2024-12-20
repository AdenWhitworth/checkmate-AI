from stockfish import Stockfish
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import chess
import random
import json
import chess.pgn

# Stockfish Skill Levels vs Human ELO (Approximation)
# -----------------------------------------------
# | Stockfish Level | Approximate Human ELO     |
# |------------------|---------------------------|
# | 1                | 800                       |
# | 2                | 900                       |
# | 3                | 1000                      |
# | 4                | 1100                      |
# | 5                | 1200                      |
# | 6                | 1300                      |
# | 7                | 1400                      |
# | 8                | 1500                      |
# | 9                | 1600                      |
# | 10               | 1700                      |
# | 11               | 1800                      |
# | 12               | 1900                      |
# | 13               | 2000                      |
# | 14               | 2100                      |
# | 15               | 2200                      |
# | 16               | 2300                      |
# | 17               | 2400                      |
# | 18               | 2500                      |
# | 19               | 2600                      |
# | 20               | 2700+ (Super Grandmaster) |
# -----------------------------------------------

# Load move mappings
with open("../Transformers/v3/models/base_transformer_full_games_15k_games_Models/move_to_id.json", "r") as f:
    moveToId = json.load(f)

with open("../Transformers/v3/models/base_transformer_full_games_15k_games_Models/id_to_move.json", "r") as f:
    idToMove = json.load(f)

# Ensure JSON keys are converted to the correct types
idToMove = {int(k): v for k, v in idToMove.items()}

# Load the trained model
model = load_model("../Transformers/v3/models/base_transformer_full_games_15k_games_Models/next_move_model.tf")
max_length = model.input_shape[1]

# Initialize Stockfish
stockfish_path = "../Stockfish/stockfish/stockfish.exe"
stockfish = Stockfish(stockfish_path)
stockfish.set_skill_level(1)  # Set Stockfish ELO level (0-20)

# Initialize PGN game
pgn_game = chess.pgn.Game()
pgn_game.headers["Event"] = "Stockfish vs Transformer"
pgn_game.headers["White"] = "Transformer Model"
pgn_game.headers["Black"] = "Stockfish"

# Set up the node for adding moves
node = pgn_game

# Function to predict the next move
def predict_next_move(model, move_history, move_to_id, id_to_move, max_length, board):
    tokenized_history = [move_to_id.get(move, 0) for move in move_history]
    padded_history = pad_sequences([tokenized_history], maxlen=max_length, padding="post")
    predictions = model.predict(padded_history, verbose=0)
    #move_probabilities = predictions[0, len(move_history)]
    move_probabilities = predictions[0]
    sorted_indices = tf.argsort(move_probabilities, direction="DESCENDING").numpy()

    for predicted_move_id in sorted_indices:
        predicted_move = id_to_move.get(predicted_move_id, "<unknown>")
        if predicted_move in [board.san(move) for move in board.legal_moves]:
            return predicted_move

    # Fallback to random move
    print("model did a random move as no legal move")
    return random.choice([board.san(move) for move in board.legal_moves])

# Initialize the chess board and move history
board = chess.Board()
move_history = []

# Play a game between the model and Stockfish
while not board.is_game_over():
    if board.turn:  # Model's turn (White if True, Black if False)
        predicted_move = predict_next_move(model, move_history, moveToId, idToMove, max_length, board)
        print(f"Model's move: {predicted_move}")
        if predicted_move in [board.san(move) for move in board.legal_moves]:
            board.push_san(predicted_move)
            move_history.append(predicted_move)
            node = node.add_variation(chess.Move.from_uci(board.peek().uci()))
        else:
            print(f"Invalid move predicted by model: {predicted_move}")
            break
    else:  # Stockfish's turn
        stockfish.set_fen_position(board.fen())  # Synchronize Stockfish with the current board
        print(f"FEN passed to Stockfish: {board.fen()}")
        stockfish_move = stockfish.get_best_move()
        print(f"Stockfish's move: {stockfish_move}")

        # Convert Stockfish move to chess.Move and validate
        stockfish_move_obj = chess.Move.from_uci(stockfish_move)
        if stockfish_move_obj in board.legal_moves:
            san_move = board.san(stockfish_move_obj)
            board.push(stockfish_move_obj)
            print("San move",san_move)
            move_history.append(san_move)
            node = node.add_variation(stockfish_move_obj)
        else:
            print(f"Invalid move from Stockfish: {stockfish_move}")
            print(f"Legal moves: {[board.san(move) for move in board.legal_moves]}")
            break

# Output the game result
result = board.result()  # Example: "1-0" (White wins), "0-1" (Black wins), "1/2-1/2" (Draw)
print(f"Game over. Result: {result}")

pgn_game.headers["Result"] = result

# Save the PGN to a file
with open("game.pgn", "w") as pgn_file:
    pgn_file.write(str(pgn_game))

# Print the PGN to the console
print("Final PGN:")
print(str(pgn_game))


