"""
Script to retrieve and display the number of logical processors (threads) on the system.

This script:
1. Uses `os.cpu_count` to determine the number of threads available on the system.
2. Prints the number of threads in a formatted manner.

Functions:
- get_logical_processors: Retrieves the number of logical processors on the system.
"""

import os

def get_logical_processors():
    """
    Retrieve the number of logical processors (threads) available on the system.

    Returns:
        int: Number of logical processors or None if the count cannot be determined.
    """
    return os.cpu_count()

if __name__ == "__main__":
    # Get and display the number of threads
    threads = get_logical_processors()
    if threads is not None:
        print(f"Number of logical processors (threads) available: {threads}")
    else:
        print("Unable to determine the number of logical processors.")
