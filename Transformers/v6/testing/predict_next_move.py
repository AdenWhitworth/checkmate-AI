"""
Chess Move Prediction and Alpha-Beta Pruning with Transformers
=============================================================

This script integrates a Transformer-based deep learning model with traditional 
chess evaluation techniques to predict optimal chess moves during the middle game phase.

**Features**:
1. **Deep Learning Predictions**:
   - Utilize a trained Transformer model to predict the best moves based on FEN positions and move sequences.
   - Incorporate centipawn (CP) and mate evaluations for move quality assessment.
2. **Alpha-Beta Pruning**:
   - Perform depth-limited search for move evaluations, leveraging the model predictions for efficiency.
   - Include a transposition table to avoid redundant computations.
3. **Evaluation Weighting**:
   - Combine probability-based predictions with model evaluations for a weighted score.
   - Prioritize decisive mate evaluations when detected.
4. **Flexibility**:
   - Support multiple prediction strategies: weighted, depth-limited, or standard.

**Usage**:
- The script predicts the best move for a given FEN position and move history.
- Users can select between `weighted`, `depth`, and `standard` assessment methods.
"""
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import chess
import chess.pgn
import chess.engine
import os

transposition_table = {}

def top_k_accuracy(y_true, y_pred, k=5):
    """
    Calculate the top-k categorical accuracy for sparse labels.

    Args:
        y_true: Ground truth labels (integer indices).
        y_pred: Predicted probabilities for each class.
        k (int): Number of top predictions to consider.

    Returns:
        tf.Tensor: Top-k accuracy as a scalar tensor.
    """
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

def init_model(model_file, move_to_idx_file, custom_objects=None):
    """
    Initialize the Transformer model and move mappings.

    Args:
        model_file (str): Path to the saved Keras model file.
        move_to_idx_file (str): Path to the JSON file containing move-to-index mapping.
        custom_objects (dict, optional): Custom objects required by the model.

    Returns:
        tuple: A tuple containing:
            - model: Loaded Transformer model.
            - idx_to_move (dict): Mapping of indices to UCI moves.
            - move_to_idx (dict): Mapping of UCI moves to indices.
    """
    with open(move_to_idx_file, "r") as f:
        move_to_idx = json.load(f)

    idx_to_move = {v: k for k, v in move_to_idx.items()}

    if custom_objects:
        model = load_model(model_file, custom_objects=custom_objects)
    else:
        model = load_model(model_file)

    # Load the trained model
    return model, idx_to_move, move_to_idx

def middle_fen_to_tensor(fen):
    """
    Convert a FEN string into a 3D tensor representation for deep learning input.

    Args:
        fen (str): FEN string representing the board state.

    Returns:
        np.ndarray: Tensor representation of the FEN with shape (8, 8, 13).
    """
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
    Convert a list of UCI moves into tensor indices based on a move-to-index mapping.

    Args:
        moves (list): List of UCI move strings.
        move_map (dict): Mapping of UCI moves to indices.

    Returns:
        list: List of move indices.
    """
    return [move_map[move] for move in moves if move in move_map]

def is_legal_move(fen, move):
    """
    Check if a given move is legal in the given FEN position.

    Args:
        fen (str): FEN string of the current board state.
        move (str): UCI move string to validate.

    Returns:
        bool: True if the move is legal, False otherwise.
    """
    board = chess.Board(fen)
    return chess.Move.from_uci(move) in board.legal_moves

def predict_middle_move_weighted_all(
    fen, moves, model, move_to_idx, idx_to_move,
    max_move_length=195, top_n=5, weight_prob=0.4, weight_eval=0.6,
    mate_eval_priority=100, max_repetition=2, repetition_penalty=0.1,
    max_cycle_length=2, cycle_penalty=0.5
):
    """
    Predict and evaluate all legal moves in a position with weighted scoring, repetition, and cycle tracking.

    Args:
        fen (str): FEN string of the current position.
        moves (list): List of previous moves in UCI format.
        model: Trained middle-game Transformer model.
        move_to_idx (dict): Mapping of UCI moves to indices.
        idx_to_move (dict): Mapping of indices to UCI moves.
        max_move_length (int): Maximum sequence length for padding.
        top_n (int): Number of top moves to consider.
        weight_prob (float): Weight for the move probability in scoring.
        weight_eval (float): Weight for the evaluation score in scoring.
        mate_eval_priority (float): Priority factor for mate evaluations.
        max_repetition (int): Number of sequential repetitions allowed before penalty.
        repetition_penalty (float): Penalty factor for repeated moves.
        max_cycle_length (int): Maximum length of cycles to detect.
        cycle_penalty (float): Penalty factor for detected cycles.

    Returns:
        list: A sorted list of top moves with evaluations.
    """
    fen_tensor = np.expand_dims(middle_fen_to_tensor(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Predict using the model
    move_pred, cp_preds, mate_preds = model.predict([fen_tensor, moves_tensor])

    # Track repeated moves in sequence
    repetitions = track_repetitions(moves, max_repetition)

    # Detect repeated move cycles
    cycles = detect_move_cycles(moves, max_cycle_length)

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
                eval_score = mate_score
                prob_weight = 0.2  # Deprioritize probability
                eval_weight = 0.8  # Prioritize evaluation
            else:
                eval_score = cp_score
                prob_weight = weight_prob
                eval_weight = weight_eval

            # Weighted score calculation
            weighted_score = prob_weight * move_pred[0][idx] + eval_weight * eval_score

            # Apply repetition penalty if the move is repeated
            if predicted_move in repetitions:
                weighted_score -= repetition_penalty * repetitions[predicted_move]

            # Apply cycle penalty if the move is part of a detected cycle
            for cycle, count in cycles.items():
                if predicted_move in cycle:
                    weighted_score -= cycle_penalty * count

            # Avoid negative scores
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
    Perform depth-limited alpha-beta pruning to find the best move.

    Args:
        game_moves (list): List of previous moves in UCI format.
        depth (int): Maximum search depth.
        alpha (float): Alpha value for pruning (best already explored option for maximizer).
        beta (float): Beta value for pruning (best already explored option for minimizer).
        is_maximizing (bool): Whether the current player is maximizing.
        model: Trained middle-game Transformer model.
        move_to_idx (dict): Mapping of UCI moves to indices.
        idx_to_move (dict): Mapping of indices to UCI moves.

    Returns:
        tuple: Best score and corresponding move in UCI format.
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

# Add this function to track repetitive moves
def track_repetitions(move_history, max_repetition=2):
    """
    Track consecutive repetitions in the move history.

    Args:
        move_history (list): List of UCI moves played so far.
        max_repetition (int): Number of times a move can repeat before being penalized.

    Returns:
        dict: A dictionary where keys are repeated moves and values are the repetition counts.
    """
    repetitions = {}
    sequence = []
    for move in reversed(move_history):
        if sequence and move != sequence[-1]:
            break
        sequence.append(move)

    for move in sequence:
        repetitions[move] = repetitions.get(move, 0) + 1

    return {move: count for move, count in repetitions.items() if count >= max_repetition}

def detect_move_cycles(move_history, max_cycle_length=2):
    """
    Detect repeated move cycles in the move history.

    Args:
        move_history (list): List of UCI moves played so far.
        max_cycle_length (int): Maximum length of cycles to detect.

    Returns:
        dict: A dictionary with detected cycles and their counts.
    """
    cycle_counts = {}
    history_len = len(move_history)

    # Check for repeated patterns of moves
    for cycle_length in range(1, max_cycle_length + 1):
        if history_len < 2 * cycle_length:
            continue

        # Extract the most recent cycle_length moves
        recent_moves = move_history[-cycle_length:]
        prior_moves = move_history[-2 * cycle_length:-cycle_length]

        if recent_moves == prior_moves:
            cycle = tuple(recent_moves)
            cycle_counts[cycle] = cycle_counts.get(cycle, 0) + 1

    return cycle_counts

def adjust_weights_by_material_count(piece_count):
    """
    Adjust weights based on the total number of pieces on the board.

    Args:
        piece_count (int): Total number of pieces on the board.

    Returns:
        tuple: (weight_prob, weight_eval) based on material count.
    """
    if piece_count < 25:
        return 0, 1
    else: 
        return 0.7, 0.3
    
def count_pieces_from_fen(fen):
    """
    Count the total number of pieces on the board based on the FEN string.

    Args:
        fen (str): FEN string of the position.

    Returns:
        int: Total number of pieces on the board.
    """
    piece_count = 0
    board_part = fen.split()[0]  # Extract the board configuration part
    for char in board_part:
        if char.isalpha():  # Count only piece characters
            piece_count += 1
    return piece_count

def predict_middle_move_weighted_priority(
    fen, moves, model, move_to_idx, idx_to_move,
    max_move_length=195, top_n=5, weight_prob=0.4, weight_eval=0.6,
    mate_eval_priority=15, king_safety_penalty=0.3
):
    fen_tensor = np.expand_dims(middle_fen_to_tensor(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Predict using the model
    move_pred, cp_preds, mate_preds = model.predict([fen_tensor, moves_tensor])

    # Normalize CP and Mate Evaluations
    def normalize_cp(cp):
        return max(min(cp / 500.0, 1), -1)

    def normalize_mate(mate):
        if abs(mate) < 0.01:
            return 0
        elif mate < 0:
            return max(-1, mate / mate_eval_priority)
        else:
            return min(1, mate / mate_eval_priority)

    # Determine game phase
    board = chess.Board(fen)
    is_endgame = len(board.piece_map()) < 14

    # Sort moves by predicted probabilities
    sorted_indices = np.argsort(move_pred[0])[::-1]
    top_moves = []

    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]
        if chess.Move.from_uci(predicted_move) in board.legal_moves:
            cp_raw = cp_preds[0][idx] * 1000.0
            mate_raw = mate_preds[0][idx]

            # Normalize evaluations
            cp_eval = normalize_cp(cp_raw)
            mate_eval = normalize_mate(mate_raw)

            # Adjust weights for aggressive play in winning positions
            if cp_eval > 1:
                weight_prob += 0.2
                weight_eval -= 0.1

            # Endgame king safety consideration
            if is_endgame:
                king_safety_penalty += 0.1

            # Blended evaluation
            blended_eval = 0.8 * mate_eval + 0.4 * cp_eval
            weighted_score = weight_prob * (move_pred[0][idx] ** 0.75) + weight_eval * blended_eval

            # Penalize king exposure
            if board.is_check():
                weighted_score -= king_safety_penalty

            # Append move details
            top_moves.append({
                "move": predicted_move,
                "probability": move_pred[0][idx],
                "cp_eval_raw": cp_raw,
                "cp_eval_normalized": cp_eval,
                "mate_eval_raw": mate_raw,
                "mate_eval_normalized": mate_eval,
                "blended_eval": blended_eval,
                "weighted_score": max(weighted_score, 0),
            })

            if len(top_moves) == top_n:
                break

    # Sort top moves
    top_moves = sorted(top_moves, key=lambda x: x["weighted_score"], reverse=True)
    best_move = top_moves[0]["move"] if top_moves else None
    return best_move, top_moves

def predict_middle_move_weighted(
    fen, moves, model, move_to_idx, idx_to_move,
    max_move_length=195, top_n=5, weight_prob=0.4, weight_eval=0.6,
    mate_eval_priority=15, max_repetition=2, repetition_penalty=0.5,
    max_cycle_length=2, cycle_penalty=2
):
    fen_tensor = np.expand_dims(middle_fen_to_tensor(fen), axis=0)
    move_indices = uci_to_tensor(moves, move_to_idx)
    moves_tensor = pad_sequences([move_indices], maxlen=max_move_length, padding="post")

    # Predict using the model
    move_pred, cp_preds, mate_preds = model.predict([fen_tensor, moves_tensor])

    # Normalize CP and Mate Evaluations
    def normalize_cp(cp):
        return max(min(cp / 500.0, 1), -1)  # Scale [-500, 500] to [-1, 1]

    def normalize_mate(mate):
        if abs(mate) < 0.01:
            return 0
        elif mate < 0:
            return max(-1, mate / mate_eval_priority)
        else:
            return min(1, mate / mate_eval_priority)

    # Track repeated moves and cycles
    repetitions = track_repetitions(moves, max_repetition)
    cycles = detect_move_cycles(moves, max_cycle_length)

    # Sort moves by predicted probabilities
    sorted_indices = np.argsort(move_pred[0])[::-1]

    # Create a chess.Board for legality checks
    board = chess.Board(fen)

    top_moves = []

    for idx in sorted_indices:
        predicted_move = idx_to_move[idx]

        # Check if the move is legal
        if chess.Move.from_uci(predicted_move) in board.legal_moves:
            # Raw values
            cp_raw = cp_preds[0][idx] * 1000.0  # Convert CP to centipawns
            mate_raw = mate_preds[0][idx]       # Mate evaluation in plies

            # Normalized values
            cp_eval = normalize_cp(cp_raw)
            mate_eval = normalize_mate(mate_raw)

            # Dynamic Blending of Evaluations
            if cp_eval > 1:  # Aggressive weighting in strong positions
                weight_eval += 0.2
                weight_prob -= 0.2  # Reduce reliance on probability

            if mate_eval == 0:
                mate_weight = 0.0
                cp_weight = 1.0 if cp_eval > 0 else 0.8
            else:
                mate_weight = 0.8 if cp_eval > 1 else 0.6
                cp_weight = 0.4 if cp_eval > 1 else 0.4

            blended_eval = mate_weight * mate_eval + cp_weight * cp_eval

            # Weighted score calculation
            scaled_blended_eval = max(blended_eval, -0.5)
            adjusted_probability = move_pred[0][idx] ** 0.75
            weighted_score = weight_prob * adjusted_probability + weight_eval * scaled_blended_eval

            # Reduce penalties in strong positions
            if cp_eval > 2:
                repetition_penalty *= 0.5
                cycle_penalty *= 0.5

            # Apply penalties for repetitions and cycles
            if predicted_move in repetitions:
                penalty_factor = 1 + repetitions[predicted_move]
                weighted_score -= repetition_penalty * penalty_factor

            for cycle, count in cycles.items():
                if predicted_move in cycle:
                    weighted_score -= cycle_penalty * count

            # Avoid negative scores
            weighted_score = max(weighted_score, 0)

            # Append to top_moves with both raw and normalized values
            top_moves.append({
                "move": predicted_move,
                "probability": move_pred[0][idx],      # Raw probability
                "cp_eval_raw": cp_raw,                # Raw CP value
                "cp_eval_normalized": cp_eval,        # Normalized CP value
                "mate_eval_raw": mate_raw,            # Raw mate value
                "mate_eval_normalized": mate_eval,    # Normalized mate value
                "blended_eval": blended_eval,         # Blended evaluation score
                "weighted_score": weighted_score,     # Final weighted score
            })

            if len(top_moves) == top_n:
                break

    # Sort the top moves by weighted score
    top_moves = sorted(top_moves, key=lambda x: x["weighted_score"], reverse=True)
    print("Top moves:", top_moves[:5])

    # Return the best move and all top moves
    best_move = top_moves[0]["move"] if top_moves else None
    return best_move, top_moves

def predict_middle_move(fen, moves, model, move_to_idx, idx_to_move, max_move_length=195, top_n=5):
    """
    Predict the best move based solely on model predictions without additional weighting.

    Args:
        fen (str): FEN string of the position.
        moves (list): List of previous moves in UCI format.
        model: Trained middle-game Transformer model.
        move_to_idx (dict): Mapping of UCI moves to indices.
        idx_to_move (dict): Mapping of indices to UCI moves.
        max_move_length (int): Maximum sequence length for padding.
        top_n (int): Number of top moves to return.

    Returns:
        tuple: Best move and list of top moves.
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
            top_moves.append({
                "move": predicted_move,
                "probability": move_pred[0][idx],
                "cp_eval": cp_preds[0][idx] * 1000.0,  # Scale back to centipawns
                "mate_eval": mate_preds[0][idx]
            })
            if len(top_moves) == top_n:
                break

    return top_moves[0]["move"], top_moves

def predict_next_move(CHECKPOINT_DIR, assessment_type, fen, move_history):
    """
    Predict the next move based on the selected assessment strategy.

    Args:
        CHECKPOINT_DIR (str): Path to the model checkpoint directory.
        assessment_type (str): Type of assessment ('weighted', 'depth', 'standard').
        fen (str): FEN string of the current position.
        move_history (list): List of past moves in UCI format.

    Returns:
        str: Best move in UCI format.
    """
    middle_model, middle_idx_to_move, middle_move_to_idx = init_model(
        os.path.join(CHECKPOINT_DIR, "model_midgame_final.h5"),
        os.path.join(CHECKPOINT_DIR, "move_to_idx.json"),
        custom_objects={"top_k_accuracy": top_k_accuracy}
    )

    if assessment_type == "weighted":
        middle_game_moves = move_history[10:]
        best_move, top_moves = predict_middle_move_weighted_priority(fen, middle_game_moves, middle_model, middle_move_to_idx, middle_idx_to_move)
    elif assessment_type == "depth":
        depth = 2
        alpha = -float("inf")
        beta = float("inf")
        is_maximizing = True
        best_score, best_move = alpha_beta_pruning(move_history, depth, alpha, beta, is_maximizing, middle_model, middle_move_to_idx, middle_idx_to_move)
    else:
        middle_game_moves = move_history[10:]
        best_move, top_moves = predict_middle_move(fen, middle_game_moves, middle_model, middle_move_to_idx, middle_idx_to_move)

    return best_move

if __name__ == "__main__":
    CHECKPOINT_DIR = "../models/checkpoints3"

    fen = "1rbqk1n1/2p5/p7/1P2nppr/2PPp2p/2P1P1PP/1B1Q1P2/R3R2K w KQkq - 0 1"
    move_history = ["d2d4", "e7e5", "g1f3", "e5e4", "c2c4", "f8b4", "b1c3", "b4c3", "b2c3", "d7d5", "d1d2", "h7h6", "f3e5", "f7f6", "e2e3", "g7g5", "a2a4", "f6f5", "g2g3", "h6h5", "f1d3", "a7a6","e1g1", "b7b5", "f1e1", "h5h4", "d3e4", "d5e4", "a4b5", "b8d7",
        "h2h3", "h8h5", "c1b2", "a8b8", "g1h1", "d7e5"]
    
    assessment_type = "depth" # Choose from "weighted", "depth", "standard"

    best_move = predict_next_move(CHECKPOINT_DIR, assessment_type, fen, move_history)

    print(f"Best transformer move: {best_move} for assessment: {assessment_type}")
