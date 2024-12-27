import os

# Get the number of logical processors (threads)
threads = os.cpu_count()
print(f"Number of threads: {threads}")