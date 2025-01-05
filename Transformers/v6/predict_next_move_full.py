import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import chess
import requests
import chess.pgn
from stockfish import Stockfish
import chess.engine

transposition_table = {}

def top_k_accuracy(y_true, y_pred, k=5):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

def init_model(model_file, move_to_idx_file, custom_objects=None):
    with open(move_to_idx_file, "r") as f:
        move_to_idx = json.load(f)

    idx_to_move = {v: k for k, v in move_to_idx.items()}

    if custom_objects:
        model = load_model(model_file, custom_objects=custom_objects)
    else:
        model = load_model(model_file)

    # Load the trained model
    return model, idx_to_move, move_to_idx

opening_model, opening_idx_to_move, opening_move_to_idx = init_model(
    "../v5/models/checkpoints3/model_final_with_outcome.h5",
    "../v5/models/checkpoints3/move_to_idx.json"
)

middle_model, middle_idx_to_move, middle_move_to_idx = init_model(
    "./models/checkpoints9/model_midgame_final.h5",
    "./models/checkpoints9/move_to_idx.json",
    custom_objects={"top_k_accuracy": top_k_accuracy}
)

# Preprocessing functions
def opening_fen_to_tensor(fen):
    board, turn, _, _, _, _ = fen.split()
    board_tensor = []
    for char in board:
        if char.isdigit():
            board_tensor.extend([0] * int(char))
        elif char.isalpha():
            board_tensor.append(ord(char))
    turn_tensor = [1] if turn == 'w' else [0]
    return np.array(board_tensor + turn_tensor, dtype=np.int32)

def middle_fen_to_tensor(fen):
    piece_map = {'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,
                 'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12}
    tensor = np.zeros((8, 8, 12), dtype=np.int32)
    board, turn = fen.split()[:2]

    row, col = 0, 0
    for char in board:
        if char.isdigit():
            col += int(char)
        elif char == '/':
            row += 1
            col = 0
        else:
            piece_idx = piece_map[char]
            tensor[row, col, piece_idx - 1] = 1
            col += 1

    turn_tensor = np.ones((8, 8, 1)) if turn == 'w' else np.zeros((8, 8, 1))
    return np.concatenate([tensor, turn_tensor], axis=-1)

def uci_to_tensor(moves, move_map):
    """
    Convert UCI move sequence to tensor representation.
    """
    return [move_map[move] for move in moves if move in move_map]

def is_legal_move(fen, move):
    """
    Check if a move is legal in the given FEN position.
    """
    board = chess.Board(fen)
    return chess.Move.from_uci(move) in board.legal_moves

# Function to predict the next move
def predict_opening_move(fen, moves, model, move_to_idx, idx_to_move, max_move_length=28):
    # Prepare FEN tensor
    fen_tensor = np.expand_dims(opening_fen_to_tensor(fen), axis=0)

    # Prepare moves tensor with padding
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Make prediction
    move_pred, outcome_pred = model.predict([fen_tensor, moves_tensor])

    # Decode next move
    sorted_indices = np.argsort(move_pred[0])[::-1]
    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]
        if is_legal_move(fen, predicted_move):
            break

    # Decode outcome prediction using updated mapping
    reverse_outcome_map = {0: "Loss", 1: "Draw", 2: "Win"}
    predicted_outcome = reverse_outcome_map[np.argmax(outcome_pred[0])]

    return predicted_move, predicted_outcome

def predict_middle_move_weighted_all(
    fen, moves, model, move_to_idx, idx_to_move, max_move_length=195, top_n=5, 
    weight_prob=0.4, weight_eval=0.6, mate_eval_priority=100
):
    fen_tensor = np.expand_dims(middle_fen_to_tensor(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Predict using the model
    move_pred, cp_preds, mate_preds = model.predict([fen_tensor, moves_tensor])

    # Sort moves by predicted probabilities
    sorted_indices = np.argsort(move_pred[0])[::-1]

    # Create a chess.Board for legality checks
    board = chess.Board(fen)

    top_moves = []

    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]

        # Check if the move is legal
        if chess.Move.from_uci(predicted_move) in board.legal_moves:
            cp_eval = cp_preds[0][idx] * 1000.0  # Scale back to centipawns
            mate_eval = mate_preds[0][idx]  # Raw mate eval in plies
            
            # Normalize evaluations
            cp_score = cp_eval / 1000.0  # Scale to [-1, 1]
            mate_score = -mate_eval / mate_eval_priority if mate_eval != 0 else 0.0
            
            # Determine final evaluation score
            if abs(mate_eval) < mate_eval_priority and mate_eval != 0:
                # Use mate_score if decisive mate is detected
                eval_score = mate_score
                prob_weight = 0.2  # Deprioritize probability
                eval_weight = 0.8  # Prioritize evaluation
            else:
                # Use cp_score otherwise
                eval_score = cp_score
                prob_weight = weight_prob
                eval_weight = weight_eval

            # Weighted score calculation
            weighted_score = prob_weight * move_pred[0][idx] + eval_weight * eval_score

            # Avoid negative scores (optional adjustment)
            weighted_score = max(weighted_score, 0)

            top_moves.append({
                "move": predicted_move,
                "probability": move_pred[0][idx],
                "cp_eval": cp_eval,
                "mate_eval": mate_eval,
                "weighted_score": weighted_score,
            })

    # Sort the top moves by weighted score
    top_moves = sorted(top_moves, key=lambda x: x["weighted_score"], reverse=True)
    return top_moves

def alpha_beta_pruning(game_moves, depth, alpha, beta, is_maximizing, model, move_to_idx, idx_to_move):
    """
    Perform depth-limited alpha-beta pruning to evaluate the best move.
    Create a fresh board instance using the game move history for each call.
    """
    # Recreate the board from game moves to avoid modifying the original
    board = chess.Board()
    for move in game_moves:
        board.push(chess.Move.from_uci(move))

    # Get the current FEN
    fen = board.fen()

    # Check transposition table
    if fen in transposition_table:
        return transposition_table[fen]

    # Check for terminal state or depth limit
    if depth == 0 or board.is_game_over():
        middle_game_moves = game_moves[10:]  # Exclude first 10 moves for midgame
        all_moves = predict_middle_move_weighted_all(fen, middle_game_moves, model, move_to_idx, idx_to_move)
        if not all_moves:
            # No moves available, return an evaluation based on maximizing player
            return (-float("inf"), None) if is_maximizing else (float("inf"), None)

        best_score = max(move["weighted_score"] for move in all_moves)
        best_move = all_moves[0]["move"]
        transposition_table[fen] = (best_score, best_move)
        return best_score, best_move.uci() if isinstance(best_move, chess.Move) else best_move

    best_score = -float("inf") if is_maximizing else float("inf")
    best_move = None

    # Predict moves using `predict_middle_move_weighted_all` for move ordering
    middle_game_moves = game_moves[10:]
    all_moves = predict_middle_move_weighted_all(fen, middle_game_moves, model, move_to_idx, idx_to_move)

    if not all_moves:
        print("Fallback to `predict_middle_move_weighted`")
        # Use a fallback to `predict_middle_move_weighted`
        best_move, top_moves = predict_middle_move_weighted(fen, middle_game_moves, model, move_to_idx, idx_to_move)
        if top_moves:
            best_score = top_moves[0]["weighted_score"]
            transposition_table[fen] = (best_score, best_move)
        return best_score, best_move.uci() if isinstance(best_move, chess.Move) else best_move

    # Iterate through all sorted moves
    for move_data in all_moves:
        move = chess.Move.from_uci(move_data["move"])
        next_game_moves = game_moves + [move.uci()]  # Use a copy of game_moves

        # Recursively evaluate the move
        score, _ = alpha_beta_pruning(
            next_game_moves, depth - 1, alpha, beta, not is_maximizing, model, move_to_idx, idx_to_move
        )

        # Update best score and move
        if is_maximizing:
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        else:
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)

        # Prune the search
        if beta <= alpha:
            break

    # Store result in transposition table
    transposition_table[fen] = (best_score, best_move.uci() if isinstance(best_move, chess.Move) else best_move)
    return best_score, best_move.uci() if isinstance(best_move, chess.Move) else best_move

def predict_middle_move_weighted(
    fen, moves, model, move_to_idx, idx_to_move, max_move_length=195, top_n=5, 
    weight_prob=0.4, weight_eval=0.6, mate_eval_priority=100
):
    """
    Predict the best move during the middle game using weighted probability and evaluation scores.
    
    Args:
        fen (str): FEN string of the position.
        moves (list): List of previous moves in UCI format.
        model: Trained middle-game model.
        move_to_idx (dict): Mapping of UCI moves to indices.
        idx_to_move (dict): Mapping of indices to UCI moves.
        max_move_length (int): Maximum number of moves for padding.
        top_n (int): Number of top moves to return.
        weight_prob (float): Base weight for the move probability in scoring.
        weight_eval (float): Base weight for the evaluation score in scoring.
        mate_eval_priority (float): Priority factor for mate evaluations.

    Returns:
        (dict, list): Best move and top moves with scores and evaluations.
    """
    fen_tensor = np.expand_dims(middle_fen_to_tensor(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Predict using the model
    move_pred, cp_preds, mate_preds = model.predict([fen_tensor, moves_tensor])

    # Sort moves by predicted probabilities
    sorted_indices = np.argsort(move_pred[0])[::-1]

    # Create a chess.Board for legality checks
    board = chess.Board(fen)

    top_moves = []

    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]

        # Check if the move is legal
        if chess.Move.from_uci(predicted_move) in board.legal_moves:
            cp_eval = cp_preds[0][idx] * 1000.0  # Scale back to centipawns
            mate_eval = mate_preds[0][idx]  # Raw mate eval in plies
            
            # Normalize evaluations
            cp_score = cp_eval / 1000.0  # Scale to [-1, 1]
            mate_score = -mate_eval / mate_eval_priority if mate_eval != 0 else 0.0
            
            # Determine final evaluation score
            if abs(mate_eval) < mate_eval_priority and mate_eval != 0:
                # Use mate_score if decisive mate is detected
                eval_score = mate_score
                prob_weight = 0.2  # Deprioritize probability
                eval_weight = 0.8  # Prioritize evaluation
            else:
                # Use cp_score otherwise
                eval_score = cp_score
                prob_weight = weight_prob
                eval_weight = weight_eval

            # Weighted score calculation
            weighted_score = prob_weight * move_pred[0][idx] + eval_weight * eval_score

            # Avoid negative scores (optional adjustment)
            weighted_score = max(weighted_score, 0)

            top_moves.append({
                "move": predicted_move,
                "probability": move_pred[0][idx],
                "cp_eval": cp_eval,
                "mate_eval": mate_eval,
                "weighted_score": weighted_score,
            })

            if len(top_moves) == top_n:
                break

    # Sort the top moves by weighted score
    top_moves = sorted(top_moves, key=lambda x: x["weighted_score"], reverse=True)
    print("Top moves: ", top_moves)
    # Return the best move and all top moves
    best_move = top_moves[0]["move"] if top_moves else None
    return best_move, top_moves

def predict_middle_move(fen, moves, model, move_to_idx, idx_to_move, max_move_length=195, top_n=5):
    fen_tensor = np.expand_dims(middle_fen_to_tensor(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Predict using the model
    move_pred, cp_preds, mate_preds = model.predict([fen_tensor, moves_tensor])

    # Sort moves by predicted probabilities
    sorted_indices = np.argsort(move_pred[0])[::-1]

    # Create a chess.Board for legality checks
    board = chess.Board(fen)

    top_moves = []

    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]

        # Check if the move is legal
        if chess.Move.from_uci(predicted_move) in board.legal_moves:
            top_moves.append({
                "move": predicted_move,
                "probability": move_pred[0][idx],
                "cp_eval": cp_preds[0][idx] * 1000.0,  # Scale back to centipawns
                "mate_eval": mate_preds[0][idx]
            })
            if len(top_moves) == top_n:
                break

    return top_moves[0]["move"], top_moves

def query_tablebase(fen):
    """
    Query the Lichess tablebase for positions with 7 or fewer pieces.
    Args:
        fen (str): FEN string of the position.

    Returns:
        dict: Tablebase response or None if query fails.
    """
    url = f"https://tablebase.lichess.ovh/standard?fen={fen}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def predict_end_move(fen):
    """
    Get the best move from the tablebase for a position with 7 or fewer pieces.
    Args:
        fen (str): FEN string of the position.

    Returns:
        str: The best move in UCI format, or None if not available.
    """
    tablebase_data = query_tablebase(fen)
    if tablebase_data and "moves" in tablebase_data:
        best_move = max(tablebase_data["moves"], key=lambda move: move.get("wdl", 0))
        return best_move["uci"]
    return None

def predict_best_move(fen, moves, opening_model, middle_model, opening_move_to_idx, opening_idx_to_move, middle_move_to_idx, middle_idx_to_move):
    """
    Predict the best move based on the game state (opening, middle, or endgame).
    
    Args:
        fen (str): FEN string of the current position.
        moves (list): List of moves made so far in the game.
        opening_model: Trained model for opening moves.
        middle_model: Trained model for middle game moves.
        opening_move_to_idx (dict): Mapping of moves to indices for the opening model.
        opening_idx_to_move (dict): Mapping of indices to moves for the opening model.
        middle_move_to_idx (dict): Mapping of moves to indices for the middle game model.
        middle_idx_to_move (dict): Mapping of indices to moves for the middle game model.
    
    Returns:
        str: Best move in UCI format.
    """
    board = chess.Board(fen)
    num_pieces = len(board.piece_map())
    
    # Determine game state
    if len(moves) <= 10:  # Opening phase
        print("Game State: Opening")
        best_move, _ = predict_opening_move(fen, moves, opening_model, opening_move_to_idx, opening_idx_to_move)
    elif num_pieces <= 7:  # Endgame phase
        print("Game State: Endgame")
        best_move = predict_end_move(fen)
    else:  # Middle game phase
        print("Game State: Middle")
        
        middle_game_moves = moves[10:]
        best_move, _ = predict_middle_move_weighted(fen, middle_game_moves, middle_model, middle_move_to_idx, middle_idx_to_move)
    
    return best_move

fen = "1rbqk1n1/2p5/p7/1P2nppr/2PPp2p/2P1P1PP/1B1Q1P2/R3R2K w KQkq - 0 1"
move_history = ["d2d4", "e7e5", "g1f3", "e5e4", "c2c4", "f8b4", "b1c3", "b4c3", "b2c3", "d7d5", "d1d2", "h7h6", "f3e5", "f7f6", "e2e3", "g7g5", "a2a4", "f6f5", "g2g3", "h6h5", "f1d3", "a7a6","e1g1", "b7b5", "f1e1", "h5h4", "d3e4", "d5e4", "a4b5", "b8d7",
                "h2h3", "h8h5", "c1b2", "a8b8", "g1h1", "d7e5"]

best_move = predict_best_move(
    fen,
    move_history,
    opening_model,
    middle_model,
    opening_move_to_idx,
    opening_idx_to_move,
    middle_move_to_idx,
    middle_idx_to_move,
)
print(f"Transformer Best Move: {best_move}")

   