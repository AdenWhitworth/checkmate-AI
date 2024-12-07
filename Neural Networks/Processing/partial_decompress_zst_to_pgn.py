import zstandard as zstd

compressed_file = "../../PGN Games/lichess_db_standard_rated_2024-09.pgn.zst"
output_file = "../../PGN Games/partial_lichess_games_150k-300k.pgn"

target_game_count = 300000  # Games to extract 
skip_game_count = 150000    # Games to skip

def is_new_game_line(line):
    """Check if a line marks the start of a new game."""
    return line.startswith("[Event ")

with open(compressed_file, "rb") as compressed:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(compressed) as reader:
        with open(output_file, "w", encoding="utf-8") as output:  # Use UTF-8 encoding
            buffer = b""
            game_count = 0
            write_count = 0
            while write_count < target_game_count:
                # Read a chunk of data
                chunk = reader.read(65536)  # 64KB chunks
                if not chunk:
                    break  # End of file

                buffer += chunk
                # Split the buffer into lines
                lines = buffer.split(b"\n")
                buffer = lines.pop()  # Keep the last partial line in the buffer

                for line in lines:
                    line_decoded = line.decode("utf-8")
                    if is_new_game_line(line_decoded):
                        game_count += 1

                    # Start writing only after skipping the first `skip_game_count` games
                    if game_count > skip_game_count:
                        output.write(line_decoded + "\n")
                        if is_new_game_line(line_decoded):
                            write_count += 1
                            if write_count >= target_game_count:
                                break
