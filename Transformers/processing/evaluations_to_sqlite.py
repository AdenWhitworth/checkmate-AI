import sqlite3
import ujson
from tqdm import tqdm

# Function to process a chunk of JSON lines
def process_chunk(chunk, db_file):
    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Prepare data for insertion
    data = []
    for line in chunk:
        try:
            fen = line["fen"]
            evals = line.get("evals", [])
            if evals:  # Check if evals is not empty
                # Find the evaluation with the highest depth
                best_eval = max(evals, key=lambda e: e["depth"])
                best_pv = best_eval.get("pvs", [])[0]  # Get the first PV

                # Check for "cp" or "mate" in the best PV
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

# Function to split the JSONL file into chunks
def split_file(file_path, chunk_size):
    with open(file_path, 'r') as f:
        chunk = []
        for i, line in enumerate(f):
            try:
                chunk.append(ujson.loads(line))  # Parse each line with ujson
            except Exception as e:
                print(f"Error parsing line {i}: {e}")
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

# Main function to preprocess the evaluations into SQLite
def preprocess_sequentially(eval_file, db_file, chunk_size=10000):
    print("Begin process:")

    # Connect to SQLite and create the table
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Ensure the table is created
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            fen TEXT PRIMARY KEY,
            eval_type TEXT,
            value INTEGER
        )
    """)
    conn.commit()

    # Confirm table creation
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations';")
    table_exists = cursor.fetchone()
    if table_exists:
        print("Table 'evaluations' exists or was successfully created.")
    else:
        print("Table 'evaluations' was not created. Exiting.")
        conn.close()
        return

    conn.close()

    # Optimize SQLite with PRAGMA settings
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")  # Speeds up by skipping sync to disk
    cursor.execute("PRAGMA journal_mode = MEMORY")  # Uses in-memory journal for faster writes
    conn.close()

    print("SQLite optimization applied.")

    # Process file and write data
    with open(eval_file, 'r') as f:
        total_lines = sum(1 for _ in f)

    print(f"Total lines to process: {total_lines}")

    # Initialize progress bar
    with tqdm(total=total_lines, desc="Processing lines") as pbar:
        for chunk in split_file(eval_file, chunk_size):
            process_chunk(chunk, db_file)  # Process each chunk
            pbar.update(len(chunk))  # Update progress bar by chunk size

    print("Data processing complete!")

    # Verify data was written to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Count rows
    cursor.execute("SELECT COUNT(*) FROM evaluations")
    row_count = cursor.fetchone()[0]
    print(f"Total rows in the database: {row_count}")

    # Display sample rows
    cursor.execute("SELECT * FROM evaluations LIMIT 5")
    rows = cursor.fetchall()
    print("Sample rows from the database:")
    for row in rows:
        print(row)

    conn.close()

# Example usage
eval_file = r"D:\checkmate_ai\lichess_db_eval\lichess_db_eval.jsonl"
db_file = r"D:\checkmate_ai\evaluations.db"
preprocess_sequentially(eval_file, db_file, chunk_size=10000)


