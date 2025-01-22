"""
Setup Script for SQLite Table

This script initializes the SQLite database and creates the `evaluations` table if it doesn't exist.
"""

import sqlite3

def setup_database(db_path):
    """
    Create the SQLite database and the `evaluations` table if they do not exist.

    Args:
        db_path (str): Path to the SQLite database.
    """
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            fen TEXT PRIMARY KEY,
            eval_type TEXT NOT NULL,
            value REAL NOT NULL
        )
    """)
    db_conn.commit()
    db_conn.close()
    print(f"Database setup completed. Table `evaluations` is ready at {db_path}.")

if __name__ == "__main__":
    # Path to your SQLite database
    db_path = r"D:\checkmate_ai\evaluations.db"

    # Set up the database
    setup_database(db_path)
