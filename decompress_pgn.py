import zstandard as zstd

# Replace with your .zst file path
compressed_file = "lichess_db_standard_rated_2024-10.pgn.zst"
decompressed_file = "lichess_games.pgn"

# Decompress the .zst file
with open(compressed_file, "rb") as compressed:
    with open(decompressed_file, "wb") as decompressed:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(compressed, decompressed)

print(f"Decompressed {compressed_file} to {decompressed_file}")
