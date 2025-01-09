"""
Interactive Chess Game with AI
==============================
This script implements an interactive chess game where the AI (playing as White) competes against a human player 
(playing as Black). The AI uses a trained model to predict the best moves during the opening and middle game phases, 
and tablebase queries for the endgame phase if applicable.

Features:
---------
1. **AI-Powered Chess Engine**:
   - The AI leverages `predict_best_move` to determine optimal moves based on the game state.
   - Separate models are used for opening and middle game phases.

2. **Interactive Gameplay**:
   - The human player inputs moves in UCI format (e.g., 'e2e4') via the console.
   - The script handles user input errors, invalid moves, and game termination gracefully.

3. **Game Phases**:
   - **Opening Phase**: Handled by an opening-specific model.
   - **Middle Game Phase**: Handled by a middle-game-specific model.
   - **Endgame Phase**: Not directly implemented in this version but can be expanded with tablebase queries.

4. **Chess Board Visualization**:
   - The script prints the chess board after every move.
"""

import chess
from predict_next_move_full_game import predict_best_move

def initialize_game():
    """
    Initialize a new chess game and set up the board.

    Returns:
        tuple: A `chess.Board` instance representing the initial board state and an empty move history list.
    """
    board = chess.Board()
    move_history = []
    print("Welcome to the Interactive Chess Game!")
    print("The AI will play as White, and you will play as Black.")
    print("You can input moves in UCI notation (e.g., 'e2e4'). Type 'exit' to quit.")
    print(board)
    return board, move_history

def ai_turn(board, move_history, CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE):
    """
    Handle the AI's turn to make a move.

    Args:
        board (chess.Board): The current chess board.
        move_history (list): List of UCI moves played so far.
        CHECKPOINT_DIR_OPENING (str): Path to the opening model checkpoints.
        CHECKPOINT_DIR_MIDDLE (str): Path to the middle game model checkpoints.

    Returns:
        bool: False if the AI cannot find a valid move, True otherwise.
    """
    print("\nAI is thinking...")
    best_move = predict_best_move(board.fen(), move_history, CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE)

    if best_move:
        move = chess.Move.from_uci(best_move)
        board.push(move)
        move_history.append(move.uci())
        print(f"AI Move (White): {best_move}")
        return True
    else:
        print("AI could not find a valid move. Exiting...")
        return False

def player_turn(board, move_history):
    """
    Handle the player's turn to make a move.

    Args:
        board (chess.Board): The current chess board.
        move_history (list): List of UCI moves played so far.

    Returns:
        bool: False if the player chooses to exit, True otherwise.
    """
    print("\nYour Turn (Black):")
    print(board)
    print(f"Current FEN: {board.fen()}")

    move_input = input("Enter your move (or type 'exit' to quit): ").strip()
    if move_input.lower() == 'exit':
        print("Game ended by user.")
        return False

    try:
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
            board.push(move)
            move_history.append(move.uci())
        else:
            print("Invalid move. Try again.")
    except ValueError:
        print("Invalid UCI format. Please enter a valid move (e.g., 'e7e5').")
    return True

def main(CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE):
    """
    Main function to run the interactive chess game.

    Args:
        CHECKPOINT_DIR_OPENING (str): Path to the opening model checkpoints.
        CHECKPOINT_DIR_MIDDLE (str): Path to the middle game model checkpoints.
    """
    board, move_history = initialize_game()

    while not board.is_game_over():
        if board.turn:  # AI's turn (White)
            if not ai_turn(board, move_history, CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE):
                break
        else:  # Player's turn (Black)
            if not player_turn(board, move_history):
                break

        if board.is_game_over():
            result = board.result()
            print(f"\nGame Over! Result: {result}")
            break

if __name__ == "__main__":
    CHECKPOINT_DIR_OPENING = "../../v5/models/checkpoints"
    CHECKPOINT_DIR_MIDDLE = "../models/checkpoints3"
    main(CHECKPOINT_DIR_OPENING, CHECKPOINT_DIR_MIDDLE)