"""
Script to extract a specified number of chess games from a Zstandard-compressed PGN file.

This script:
1. Decompresses a `.zst` file containing PGN data.
2. Extracts a target number of games, optionally skipping a specified number of games.
3. Writes the extracted games to an output file.

Functions:
- is_new_game_line: Checks if a line marks the start of a new chess game.
- extract_games: Extracts chess games from a compressed PGN file.

Requirements:
- python-zstandard library

Usage:
- Update `compressed_file`, `output_file`, `target_game_count`, and `skip_game_count` as needed.
- Run the script to extract and save the specified number of games.
"""

import zstandard as zstd

def is_new_game_line(line):
    """
    Check if a line marks the start of a new chess game in a PGN file.

    Args:
        line (str): A single line from the PGN file.

    Returns:
        bool: True if the line marks the start of a new game, False otherwise.
    """
    return line.startswith("[Event ")


def extract_games(compressed_file, output_file, target_game_count, skip_game_count=0):
    """
    Extracts a specified number of chess games from a compressed PGN file.

    Args:
        compressed_file (str): Path to the Zstandard-compressed PGN file.
        output_file (str): Path to save the extracted games in PGN format.
        target_game_count (int): Number of games to extract.
        skip_game_count (int): Number of games to skip before extraction (default is 0).
    """
    print(f"Starting extraction from {compressed_file}")
    print(f"Target: {target_game_count} games, Skip: {skip_game_count} games")
    game_count = 0
    write_count = 0

    with open(compressed_file, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            with open(output_file, "w", encoding="utf-8") as output:
                buffer = b""
                while write_count < target_game_count:
                    chunk = reader.read(65536)  # Read 64KB chunks
                    if not chunk:
                        break  # End of file

                    buffer += chunk
                    lines = buffer.split(b"\n")
                    buffer = lines.pop()  # Keep the last partial line for the next chunk

                    for line in lines:
                        line_decoded = line.decode("utf-8")
                        if is_new_game_line(line_decoded):
                            game_count += 1

                        # Write only after skipping the specified number of games
                        if game_count > skip_game_count:
                            output.write(line_decoded + "\n")
                            if is_new_game_line(line_decoded):
                                write_count += 1
                                if write_count >= target_game_count:
                                    print(f"Extraction complete: {write_count} games written.")
                                    return

    print(f"Extraction finished. {write_count} games written to {output_file}.")

if __name__ == "__main__":
    # Input and output file paths
    compressed_file = "../zst/lichess_db_standard_rated_2024-09.pgn.zst"
    output_file = "../pgns/partial_lichess_games_500k.pgn"

    # Extraction parameters
    target_game_count = 500000  # Number of games to extract
    skip_game_count = 0         # Number of games to skip

    # Extract games
    extract_games(compressed_file, output_file, target_game_count, skip_game_count)