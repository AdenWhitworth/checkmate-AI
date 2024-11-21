import numpy as np

def fen_to_matrix(fen):
    piece_map = {
        "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,  # Black pieces
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6        # White pieces
    }
    rows = fen.split(" ")[0].split("/")  # First part of FEN is board layout
    board_matrix = []
    
    for row in rows:
        row_array = []
        for char in row:
            if char.isdigit():
                row_array.extend([0] * int(char))  # Empty squares
            else:
                row_array.append(piece_map[char])  # Piece
        board_matrix.append(row_array)
    
    return np.array(board_matrix, dtype=np.int8)

# Example usage
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
print(fen_to_matrix(fen))
