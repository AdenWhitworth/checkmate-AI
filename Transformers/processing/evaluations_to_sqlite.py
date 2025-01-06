"""
Script to preprocess chess evaluations from a JSONL file into an SQLite database.

This script:
1. Reads chess evaluations from a JSONL file.
2. Processes each FEN with its evaluations to extract the best evaluation.
3. Stores the processed data in an SQLite database.
4. Optimizes SQLite for efficient writes.

Functions:
- process_chunk: Processes a chunk of JSON lines and inserts valid data into SQLite.
- split_file: Splits a JSONL file into manageable chunks for processing.
- preprocess_sequentially: Main function to orchestrate the preprocessing pipeline.

Requirements:
- tqdm for progress bar visualization
- ujson for faster JSON parsing
- SQLite database must be accessible

Usage:
- Update `eval_file` and `db_file` paths as needed.
- Run the script to process the JSONL file into an SQLite database.
"""

import sqlite3
import ujson
from tqdm import tqdm


def process_chunk(chunk, db_file):
    """
    Process a chunk of JSON lines and insert valid evaluations into SQLite.

    Args:
        chunk (list): List of JSON-parsed lines from the JSONL file.
        db_file (str): Path to the SQLite database file.

    Returns:
        None
    """
    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    data = []
    for line in chunk:
        try:
            fen = line["fen"]
            evals = line.get("evals", [])
            if evals:
                # Find the evaluation with the highest depth
                best_eval = max(evals, key=lambda e: e["depth"])
                best_pv = best_eval.get("pvs", [])[0]

                # Extract evaluation type and value
                if "mate" in best_pv:
                    data.append((fen, "mate", best_pv["mate"]))
                elif "cp" in best_pv:
                    data.append((fen, "cp", best_pv["cp"]))
                else:
                    print(f"No valid evaluation found in 'pvs' for FEN: {fen}")
            else:
                print(f"No evaluations found for FEN: {fen}")
        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")

    # Insert data into SQLite
    if data:
        cursor.executemany("INSERT OR IGNORE INTO evaluations VALUES (?, ?, ?)", data)
        conn.commit()
        print(f"Inserted {len(data)} rows into the database.")
    else:
        print("No valid data in this chunk.")

    conn.close()


def split_file(file_path, chunk_size):
    """
    Split a JSONL file into manageable chunks.

    Args:
        file_path (str): Path to the JSONL file.
        chunk_size (int): Number of lines per chunk.

    Yields:
        list: A chunk of JSON-parsed lines.
    """
    with open(file_path, 'r') as f:
        chunk = []
        for i, line in enumerate(f):
            try:
                chunk.append(ujson.loads(line))
            except Exception as e:
                print(f"Error parsing line {i}: {e}")
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def preprocess_sequentially(eval_file, db_file, chunk_size=10000):
    """
    Preprocess a JSONL file into an SQLite database.

    Args:
        eval_file (str): Path to the JSONL file containing evaluations.
        db_file (str): Path to the SQLite database file.
        chunk_size (int): Number of lines to process per chunk.

    Returns:
        None
    """
    print("Begin process:")

    # Connect to SQLite and create the table
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            fen TEXT PRIMARY KEY,
            eval_type TEXT,
            value INTEGER
        )
    """)
    conn.commit()

    if cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations';").fetchone():
        print("Table 'evaluations' exists or was successfully created.")
    else:
        print("Table 'evaluations' was not created. Exiting.")
        conn.close()
        return

    conn.close()

    # Optimize SQLite for faster writes
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    conn.close()
    print("SQLite optimization applied.")

    # Count total lines for progress bar
    with open(eval_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines to process: {total_lines}")

    # Process file and write data
    with tqdm(total=total_lines, desc="Processing lines") as pbar:
        for chunk in split_file(eval_file, chunk_size):
            process_chunk(chunk, db_file)
            pbar.update(len(chunk))

    print("Data processing complete!")

    # Verify data in SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM evaluations")
    print(f"Total rows in the database: {cursor.fetchone()[0]}")

    print("Sample rows from the database:")
    for row in cursor.execute("SELECT * FROM evaluations LIMIT 5").fetchall():
        print(row)

    conn.close()

if __name__ == "__main__":
    eval_file = r"D:\checkmate_ai\lichess_db_eval\lichess_db_eval.jsonl"
    db_file = r"D:\checkmate_ai\evaluations.db"
    preprocess_sequentially(eval_file, db_file, chunk_size=10000)
