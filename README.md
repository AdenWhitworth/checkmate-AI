<img width="80" src="https://github.com/AdenWhitworth/checkmate_Front-End/raw/master/src/Images/King%20Logo%20Black.svg" alt="Checkmate Logo">

# Checkmate AI

Welcome to **Checkmate AI**, the machine learning-powered chess engine driving the AI chess bot for Checkmate. Designed to simulate various ELO skill levels, this bot demonstrates advanced chess logic, adapts to different player abilities, and competes with the goal of mastering the game.

## Table of Contents
- [Overview](#overview)
- [Checkmate Demo](#checkmate-demo)
  - [Test User Credentials](#test-user-credentials)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Hardware](#hardware)
  - [Installation](#installation)
  - [Dataset](#dataset)
- [Chess Bot Creation](#chess-bot-creation)
  - [Model Architecture](#model-architecture)
  - [Current Model](#current-model)
    - [Steps to Reproduce](#steps-to-reproduce)
    - [Individual Model Creation](#individual-model-creation)
  - [Inference Process](#inference-process)
  - [Previous Models](#previous-models)
- [Performance Summary](#performance-summary)
- [Challenges & Learnings](#challenges--learnings)
- [Future Features](#future-features)
- [Contributing](#contributing)
- [License](#license)

## Overview

**Checkmate** is a real-time chess platform where players can compete against live opponents, chat in-game, track their rankings, complete interactive puzzles, and practice with human-like AI bots. The AI bot is built using TensorFlow in Python, processing games to train and test models. The trained models are exported in ONNX format for seamless integration into the Checkmate backend.

## Checkmate Demo

The Checkmate application is live and can be accessed here: [Checkmate Demo](https://checkmateplay.com). You can explore all features of the game, including real-time gameplay, puzzles, chat, and rankings.

<img width="600" src="https://github.com/AdenWhitworth/aden_whitworth_portfolio/raw/master/src/Images/chess_demo.png" alt="Checkmate Demo">

### Test User Credentials

Try out the app using the following demo accounts:

- **Emails:** demo1@gmail.com & demo2@gmail.com
- **Password:** PortfolioDemo1!

>**Note**: You can even play against yourself by opening the application in two separate browser windows.

## Features

- **PGN Processing**: Converts PGN game data into structured inputs for DNN, CNN, and transformer models, ensuring robust training and evaluation.
- **Move Validation**: Incorporates logic to validate and predict only legal chess moves, enhancing gameplay authenticity.
- **ELO-Specific Bots**: Offers finely-tuned bots trained for different ELO skill levels, providing a tailored challenge for players of all abilities.
- **Realistic Gameplay**: Models deliver high accuracy and low loss, simulating human-like decision-making and advanced chess strategies.
- **Seamless Backend Integration**: Exports trained models in ONNX format for smooth deployment and integration into the Checkmate backend.
- **Stockfish Evaluation**: Leverages Stockfish for benchmarking ELO performance and evaluating game moves, ensuring accurate assessments and consistent model refinement.
- **GPU-Accelerated Training**: Optimizes TensorFlow model training using NVIDIA CUDA support, significantly reducing training time and enabling the handling of complex datasets efficiently.

## Technologies Used

- **Chess**: A python library for validating and manipulating chess moves, ensuring the legality of each action during gameplay.
- **Keras**: A high-level neural network API that simplifies the building and training of machine learning models, used here for rapid development of DNN and CNN models.
- **Matplotlib**: A Python library for creating visualizations, such as accuracy and loss graphs, to analyze model performance.
- **NumPy**: A fundamental Python library for numerical computations, particularly useful for manipulating game data and creating training datasets.
- **ONNX**: (Open Neural Network Exchange) A format for exporting and integrating machine learning models across different frameworks, enabling smooth backend deployment.
- **Pandas**: A Python library for data manipulation and analysis, used to preprocess and structure PGN game data.
- **Python**: The primary programming language for the project, powering all data processing, model training, and backend integration.
- **Scikit-learn (SKLearn)**: A machine learning library for preprocessing data, evaluating models, and implementing supplementary algorithms like scaling and validation.
- **TensorFlow**: An end-to-end machine learning framework used for building, training, and fine-tuning AI models, including transformers and neural networks.
- **Linchess Database**: Open-source database containing millions of chess games, used to train the bots chess gameplay at varying skill levels.
- **Cudatoolkit**: A toolkit that provides libraries and tools for GPU-accelerated computing, enabling efficient execution of TensorFlow models on NVIDIA GPUs.
- **CUDNN**: NVIDIA’s CUDA Deep Neural Network library, used to optimize the performance of deep learning frameworks like TensorFlow during model training and inference.
- **Conda**: A versatile package manager that simplifies the environment setup and dependency management for Python libraries used in the project.
- **Tqdm**: A Python library for generating progress bars, providing real-time feedback during data processing and model training.
- **Onnxoptimizer**: A library for optimizing ONNX models by reducing size and improving inference speed without compromising accuracy.
- **Onnxruntime**: A high-performance runtime for executing ONNX models, enabling efficient inference of trained models during gameplay.
- **Tf2onnx**: A conversion tool that seamlessly exports TensorFlow models into the ONNX format, facilitating integration with the Checkmate backend.

## Getting Started

Follow the instructions below to set up the project on your local machine.

### Prerequisites

Ensure the following are installed:

- **Python** (v3.6 or higher, installed via Conda or standalone)
- **VS Code** (or any preferred IDE)
- **Conda** (24.11.1 or higher)
- **Lichess Standard Chess Games** (minimum 1 million games)
- **Lichess Evaluations Database**
- **Lichess Openings Dataset**
- **Git** (command-line tool for cloning repositories)

### Hardware

- **CPU Requirements**: Use a high-performance CPU with multiple cores and threads. This repository was built using a 13th Gen Intel i7 processor, ideal for tasks like Stockfish preprocessing and training.
- **GPU Requirements**: A dedicated GPU (e.g., NVIDIA RTX 4060) is recommended for TensorFlow model training to significantly reduce training time. Training can be performed on a CPU, but expect much longer runtimes. While this repository is optimized for NVIDIA GPUs using CUDA, training can also be performed on CPUs or other compatible GPUs with adjusted runtimes.
- **Storage Requirements**: Ensure at least 350GB of SSD storage for the project files, including Lichess game and evaluation databases, SQL evaluation tablebases, and annotated JSON files. Fast read/write speeds are crucial for efficient preprocessing. For example, for this repository a SK Hynix Platinum P41 2TB PCIe NVMe Gen4 M.2 SSD with up to 7,000MB/s read/write speeds was used.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AdenWhitworth/checkmate-AI.git
   ```
2. **Set up Conda environments**:

The project uses two Conda environments to manage dependencies:
- `tf_gpu`: For training and inference tasks.
- `tf_onnx_env`: For ONNX model conversion.

Create and activate the environments from the provided .yml files:

- For `tf_gpu` (training and inference):
  ```bash
  conda env create -f environment_gpu.yml
  conda activate tf_gpu
  ```

- For `tf_onnx_env` (ONNX conversion):
  ```bash
  conda env create -f environment_onnx.yml
  conda activate tf_onnx_env
  ```
3. **Verify the installations**:

Ensure the environments were set up correctly by checking the installed packages:
  ```bash
  conda list
  ```
4. **Switch between environments**:

Use the following commands to switch between environments as needed:
- Activate `tf_gpu`:
  ```bash
  conda activate tf_gpu
  ```
- Activate `tf_onnx_env`:
  ```bash
  conda activate tf_onnx_env
  ```

5. **Optional - Update dependencies**:
If you modify the environment or need to update dependencies, export the updated environment with:
  ```bash
  conda env export > environment_gpu.yml
  conda env export > environment_onnx.yml
  ```

6. **Optional - Verify GPU**:
If you plan to use a GPU, then run the following script to confirm that TensorFlow recognizes the available GPU. This ensures that TensorFlow can detect and utilize the GPU for accelerated training.
  ```bash
  cd "checkmate-AI\Hardware Checks"
  python check_gpu.py
  ```
7. **Optional - Verify CPU Threads**:
If you plan to use a stockfish, then run the following script to confirm the CPU's available thread count. This ensures your CPU has enough threads to optimize Stockfish preprocessing.
  ```bash
  cd "checkmate-AI\Hardware Checks"
  python check_cpu_threads.py
  ```

### Dataset

1. **Download Chess Games**:
Obtain publicly available chess games in PGN format from the [Lichess Standard Chess Database](https://database.lichess.org/#standard_games).
    - Approximate Size: 28GB for a full months dataset in zst format
    - Note: Processing the zst into PGN requires additional space, but can be adjusted based on the number of games you want to process and store. 1 million processed games to PGN required an approximate 1GB of additional storage. 

2. **Download Evaluations Database**:
Save Stockfish chessboard evaluations (174 million positions) in JSON format from the [Lichess Evaluations Database](https://database.lichess.org/#evals).
    - Approximate Size: 39GB

3. **Download Stockfish Binary**:
Install the Stockfish binary for your operating system from [Stockfish 17](https://stockfishchess.org/download/). Ensure the binary is added to your system’s PATH for use in preprocessing.

4. **Download Opening Book**:
Obtain the comprehensive database of known chess opening moves from the [Lichess Chess Openings](https://www.kaggle.com/datasets/lichess/chess-openings).

## Chess Bot Creation

Discover the step-by-step process behind the development of the Checkmate AI bot, from data preparation to advanced model architectures.

### Model Architecture

The bot's design evolved through iterative improvements, starting with simple models and gradually increasing in complexity:

1. **Deep Neural Network (DNN) (V1)**:
    - Served as the baseline model to understand fundamental chess patterns and evaluate the feasibility of training on the dataset.
2. **Convolutional Neural Network (CNN) (V2)**:
    - Introduced spatial reasoning to recognize patterns on the chessboard.
    - Unified input representation for improved scalability.
3. **CNN Unified** (V3): 
    - Unified input representation to include player elos
4. **CNN with Legal Moves (V4)**:
    - Enhanced prediction accuracy by filtering moves to only valid chess actions.
    - Incorporated chess-specific logic into the training process.
5. **Base Transformer (V1)**:
    - Applied a transformer architecture to improve sequence understanding and decision-making across multiple moves.
6. **Fine-Tuned Transformer (V1)**:
    - Customized the transformer model for specific ELO ranges (<1000, 1000–1500, etc.), ensuring tailored gameplay for different skill levels.
7. **Game Phase Transformer (V2)**:
    - Integrated features representing the opening, middle, and endgame phases.
    - Included a partial game generator to predict the next move in a given game phase.
8. **ELO-Normalized Transformer (V3)**:
    - Applied normalization techniques to account for ELO ranges through weight adjustments.
    - Predicted the next move in the game using normalized training data.
9. **Policy & Value Transformer (V4)**:
    - Combined policy evaluation of the next move with value evaluation based on Stockfish-annotated centipawn and mate scores.
    - Incorporated move history in the second iteration for deeper contextual understanding.
10. **Opening Transformer (V5)**:
    - Trained on a Kaggle openings dataset to predict moves and outcomes for up to 10 moves in the opening phase. (Version 5)
11. **Middle Game Transformer (V6)**:
    - Focused on policy and value evaluation for all legal moves at the current FEN position.
    - Used grandmaster moves as targets and incorporated centipawn and mate evaluations for middle-game scenarios. (Version 6)

### Current Model

#### **Overview**:
The current implementation consists of two specialized transformer models: the Opening Phase Transformer and the Midgame Phase Transformer, each designed to address specific challenges and requirements for their respective game phases. For the endgame phase, the Lichess 7 piece table base is used. 

#### **Steps to Reproduce**:

Follow these steps to replicate the training, evaluation, and testing of the current models.

1. **Setup Environment**
Ensure the following prerequisites are available and configured:
  - Install all dependencies listed in the Getting Started section.
  - Download and configure required datasets (e.g., PGN games, stockfish binary, opening book).
  - Verify the environment is functioning correctly.

2. **Preprocess Data**
  a.  Convert the `.zst` games files into a `.pgn` file
  This step decompresses raw `.zst` game files into a `.pgn` format.
  ```bash
  cd "checkmate-AI/PGN Games/utility"
  # Update the target_game_count and skip_game_count as desired.
  # Example: target_game_count = 150,000 and skip_game_count = 0
  python partial_decompress_zst_to_pgn.py
  ```
  b. Filter PGN games by ELO and Termination
  Filter games based on player ratings (ELO) and termination conditions.
  ```bash
  cd "checkmate-AI/PGN Games/utility"
  # Specify the path to the `.pgn` file created in Step 2a.
  python filter_pgn_by_elo_game_termination.py
  ```
  c. Annotate Games with an Openings Book
  Enrich games with data from a chess opening book to focus on the opening phase.
  ```bash
  cd "checkmate-AI/Transformers/v5/processing"
  # Provide paths to the filtered `.pgn` file from Step 2b and the downloaded opening book.
  python opening_processing.py
  ```
  d. Create an SQLite Database for Stockfish Caching
  Set up a database to cache Stockfish evaluations and reduce redundant calculations.
  ```bash
  cd "checkmate-AI/Transformers/processing"
  # Note: Keep track of the output path for use in the next step.
  python setup_sqlite.py
  ```
  e. Annotate Game FENs with Stockfish Evaluations
  Use Stockfish to annotate FEN positions with evaluation scores.
  ```bash
  cd "checkmate-AI/Transformers/processing"
  # Provide paths to the filtered `.pgn` file from Step 2b and the SQLite database from Step 2d.
  # Note: This script is resource-intensive. Annotating 3,200 games can take ~48 hours.
  python annotate_pgn_evals.py
  ```
3.  **Train the Models**
   a. Train the Opening Phase Transformer
  Train the model specifically for the opening phase using annotated data.
   ```bash
  cd "checkmate-AI/Transformers/v5/training"
  # Provide the path to the annotated opening data from Step 2c.
  python opening_transformer.py
  ```
   b. Train the Midgame Phase Transformer
   Train the model for midgame strategies using Stockfish-annotated data.
   ```bash
  cd "checkmate-AI/Transformers/v6/training"
  # Provide the path to the annotated middle game data from Step 2e.
  python middle_transformer.py
  ```
4. **Evaluate and Test the Models**
  a. Predict the Next Best Opening Move
  Evaluate the opening model on a specific position.
  ```bash
  cd "checkmate-AI/Transformers/v5/testing"
  # Specify the path to the trained opening model from Step 3a.
  python predict_next_move.py
  ```
  b. Predict the Next Best Midgame Move
  Evaluate the midgame model on a specific position.
  ```bash
  cd "checkmate-AI/Transformers/v6/testing"
  # Specify the path to the trained midgame model from Step 3b.
  python predict_next_move.py
  ```
  c. Predict the Next Best Full Game Move
  Combine the opening and midgame models with the 7-piece endgame tablebase for full-game inference.
  ```bash
  cd "checkmate-AI/Transformers/v6/testing"
  # Provide paths to the trained opening model (Step 3a) and midgame model (Step 3b).
  python predict_next_move_full_game.py
  ```
  d. Play Against the Full Game Model in Terminal
  Engage in a terminal-based chess game against the full-game model.
  ```bash
  cd "checkmate-AI/Transformers/v6/testing"
  # Specify paths to the opening model (Step 3a) and midgame model (Step 3b).
  python terminal_vs_transformer.py
  ```
  e. Simulate Stockfish vs. Full Game Model
  Pit Stockfish against the full-game model for benchmarking.
  ```bash
  cd "checkmate-AI/Transformers/v6/testing"
  # Provide paths to the opening model (Step 3a), midgame model (Step 3b), and Stockfish binary.
  python stockfish_vs_transformer_v5_v6.py
  ```
  f. Simulate V1 vs. V5/V6 Transformer Models
  Test the progression of model versions by simulating a game between the V1 and V5/V6 models.
  ```bash
  cd "checkmate-AI/Transformers/v6/testing"
  # Provide paths to the opening model (Step 3a) and midgame model (Step 3b).
  python transformer_v1_vs_transformer_v6.py
  ```

---

#### **Individual Model Creation**

The chess AI system uses a phase-specific strategy to improve accuracy and efficiency, dividing the game into three phases: opening, middle game, and endgame. Each phase is addressed by a tailored approach: transformer models for the opening and middle game phases, and a tablebase for the endgame. Below are detailed descriptions of these components and how they contribute to the overall system.

##### **Opening Phase Transformer**:

The Opening Phase Transformer focused on the initial stage of chess games, leveraging annotated opening book data and game outcomes to specialize in early-game strategies. This phase-specific model trained on openings provided a clearer focus on common patterns in chess openings.

- **Key Features**:
  - **Opening Phase Data Preparation**:
      - Processes games with FEN positions, UCI move sequences, and game outcomes annotated with ECO codes and opening names.
      - Labels moves with a predefined opening book to provide phase-specific context.
  - **Game Outcome Integration**:
      - Includes game outcomes (-1 for loss, 0 for draw, 1 for win) to train the model to predict positional strength in addition to next moves.
  - **Dual Outputs**:
      - Simultaneously predicts the next move in the opening sequence and the overall game outcome.
  - **Transformer Model Design**:
      - Combines FEN encoding and move sequence embeddings with a Transformer-based architecture for contextual predictions.
  - **Preprocessing Pipeline**:
      - Converts FEN strings into numeric tensors representing the board state.
      - Encodes UCI moves into sequences of token indices with a custom move-to-index mapping.
      - Pads sequences for uniform input lengths.

- **Data Preparation**:
  - **FEN Encoding**:
      - Converts chessboard states into tensors with numeric representations for pieces, turn, castling rights, and en passant information.
  - **Move Tokenization**:
      - Encodes UCI moves using a move-to-index map, with padding for sequences shorter than the maximum length.
  - **Outcome Labeling**:
      - Maps game outcomes into discrete classes: Loss (0), Draw (1), and Win (2).
  - **Dataset Splitting**:
      - Splits data into 80% training and 20% validation sets for model evaluation.

- **Model Architecture**:
  - **Inputs**:
      - **FEN Input**: Encodes the current board state.
      - **Move Sequence Input**: Encodes the sequence of prior moves.
  - **Embedding and Attention Layers**:
      - Embeds FEN and move sequences into dense vectors.
      - Uses multi-head attention to capture positional and sequential dependencies.
  - **Outputs**:
      - **Next Move Prediction**: Outputs a probability distribution over all legal moves.
      - **Game Outcome Prediction**: Outputs probabilities for game outcomes (win, draw, loss).
  - **Loss and Metrics**:
      - Tracks total loss, next move cross-entropy loss, and outcome classification loss.
      - Measures accuracy for next move predictions and game outcomes.

- **Training and Results**:
  - The model was trained for 50 epochs with early stopping and learning rate scheduling.
  - Achieved the following results:

    | Metric                    | Training Value | Validation Value |
    |---------------------------|----------------|------------------|
    | **Loss**                  | 2.5458         | 2.5144           |
    | **Next Move Loss**        | 1.6852         | 1.6496           |
    | **Outcome Loss**          | 0.8606         | 0.8648           |
    | **Next Move Accuracy**    | 46.22%         | 46.98%           |
    | **Outcome Accuracy**      | 50.14%         | 49.36%           |

- **Insights and Limitations**:
  - **Improvements**:
      - **Opening Phase Context**: Restricting the data to the opening phase allowed the model to focus on specific patterns and strategies common in early-game positions.
      - **Dual-Output Learning**: Simultaneous training for move and outcome predictions enhanced the model's understanding of positional evaluations.
      - **Efficient Data Handling**: The preprocessing pipeline effectively managed large datasets with uniform labeling and padding.
  - **Challenges**:
      - **Game Outcome Prediction**: While moderately accurate, the model's performance on outcome prediction could be improved by incorporating more advanced evaluation features (e.g., Stockfish CP/mate evaluations).
      - **Limited Scope**: Focusing only on the opening phase may limit generalizability to middle and endgame scenarios.
  - **Key Takeaways**:
      - Phase-specific models like this opening-focused one can significantly improve prediction accuracy for specific portions of the game.
      - Future work could extend the approach to middle and endgame phases, ensuring each phase benefits from tailored training data and objectives.
      - Incorporating legal move evaluations and Stockfish analysis could further refine predictions and improve model performance in real-game scenarios.

##### **Midgame Phase Transformer**:

The Midgame Phase Transformer specialized in predicting moves and evaluations during the midgame, where strategic complexity increases. By incorporating legal moves and evaluations for every board state, this model emphasized contextual understanding of the game.

- **Key Features**:
  - **Enhanced FEN Encoding**:
      - Converts FEN strings into spatial tensors, representing board state and game context with turn information.
  - **Legal Move Integration**:
      - Incorporates all legal moves from the given board state, with centipawn (CP) and mate evaluations for each move.
  - **Move History and Predictions**:
      - Processes sequences of moves up to the current position to predict the next move.
  - **Multi-Output Model**:
      - Simultaneously predicts:
          - The next move (classification).
          - CP evaluations for all legal moves (regression).
          - Mate evaluations for all legal moves (regression).
  - **Transformer-Based Architecture**:
      - Combines convolutional neural networks (CNNs) for spatial representation and attention mechanisms for sequential move analysis.

- **Data Preparation**:
  - **FEN to Tensor**:
      - Encodes FEN into an 8x8x13 tensor, capturing piece locations, turn, and castling rights.
  - **Move Encoding**:
      - Maps UCI moves to indices using a custom move-to-index dictionary.
  - **Legal Move Evaluations**:
      - Normalizes CP evaluations to a [-1, 1] range and retains mate evaluations as-is.
  - **Dataset Splitting**:
      - Splits data into 80% training and 20% validation sets.

- **Model Architecture**:
  - **Inputs**:
      - **FEN Input**: Encodes the current board state as a spatial tensor.
      - **Move Sequence Input**: Encodes prior moves leading to the current position.
  - **Embeddings**:
      - CNN extracts spatial features from FEN input.
      - Move sequences are embedded and processed using attention mechanisms.
  - **Outputs**:
      - **Next Move Prediction**: Softmax output over all possible moves.
      - **CP Evaluation Prediction**: Linear output for normalized centipawn scores.
      - **Mate Evaluation Prediction**: Linear output for mate evaluations.
  - **Loss and Metrics**:
      - Tracks loss and accuracy for next move prediction.
      - Measures mean absolute error (MAE) for CP and mate evaluations.

- **Training and Results**:
  - The model was trained across three checkpoints with increasing dataset sizes.

    | Checkpoint | Loss   | Next Move Loss | CP Loss | Mate Loss | Next Move Accuracy | Top-k Accuracy | CP MAE  | Mate MAE |
    |------------|--------|----------------|---------|-----------|--------------------|----------------|---------|----------|
    | **1 (700 games)**  | 4.8450 | 4.8378 | 0.0032  | 4.06e-05  | 11.42%            | 27.13%         | 0.0101  | 0.0015   |
    | **2 (2200 games)** | 4.2049 | 4.1999 | 0.0031  | 1.51e-05  | 15.94%            | 37.13%         | 0.0095  | 0.0005   |
    | **3 (3200 games)** | 4.1148 | 4.1104 | 0.0030  | 8.99e-06  | 15.83%            | 38.17%         | 0.0091  | 0.0004   |

- **Insights and Limitations**:
  - **Improvements**:
      - **Enhanced Context Understanding**: Combining FEN, legal moves, and evaluations improved the model's contextual predictions.
      - **Multi-Task Learning**: Simultaneous training on move prediction and evaluations led to better performance across all tasks.
      - **Scalable Training**: Incremental dataset increases significantly improved next move accuracy and top-k accuracy.
  - **Challenges**:
      - **Low Top-1 Accuracy**: Predicting the exact next move remains challenging due to the large move space and complex positional evaluations.
      - **Evaluation Noise**: Variability in CP and mate evaluations across different scenarios may introduce noise in predictions.
- **Key Takeaways**:
    - **Multi-Task Learning Potential**: The midgame model highlights the effectiveness of multi-task learning for chess prediction tasks. By incorporating legal moves and evaluations, the model improves contextual understanding, making it better suited for complex midgame scenarios.
    - **Dataset Size Impact**: The accuracy and top-k performance of the model improved significantly as the number of preprocessed games increased. However, only 3,200 games were used due to the high preprocessing overhead (~10 hours per 1,000 games). Future work could explore processing larger datasets (e.g., 10,000 games) to evaluate the impact of dataset size on accuracy and convergence.

##### **Endgame Tablebase**
The endgame phase utilizes the Lichess 7-piece tablebase to ensure optimal play when only seven or fewer pieces remain on the board. This approach eliminates the need to train a specific endgame model, as tablebases provide perfect information about every possible position within their scope.

- **Key Features**:
  - **Optimal Endgame Play**:
    - Tablebases contain precomputed outcomes for every legal chess position with seven or fewer pieces, ensuring perfect play.
    - Outcomes include win, draw, or loss information and the best possible moves for each scenario.
  - **Efficient Integration**:
    - Instead of training a model, the system queries the Lichess tablebase API during inference to retrieve move recommendations.
    - This reduces computational overhead while guaranteeing accuracy.
  - **Dynamic Decision Making**:
    - If a tablebase position indicates a win, the model plays moves to minimize the number of turns to victory.
    - In draw scenarios, the tablebase ensures moves that maximize the chance of achieving a stalemate.
- **Implementation Details**:
  - **API Integration**:
    - The system sends FEN strings of the current position to the Lichess Tablebase API.
    - The API returns a list of legal moves with associated win-draw-loss values.
  - **Best Move Selection**:
    - The move with the highest win probability (or lowest loss probability) is selected.
  - **Fallback Handling**:
    - For rare cases where API access fails, the system defaults to a conservative move strategy to avoid blunders.
- **Advantages**:
  - Guarantees perfect play in endgame scenarios.
  - Reduces training time and model complexity by offloading endgame decisions to a precomputed resource.
  - Ensures scalability as tablebases extend to larger datasets or pieces.
- **Limitations**:
  - Requires an internet connection to access the Lichess API unless the tablebase is stored locally (storage requirements: 17TB for 7-piece tablebase).
  - Limited to positions with seven or fewer pieces, requiring fallback strategies for more complex positions.
  - By integrating the tablebase for endgame decisions, the system leverages the best available resources to ensure optimal performance while focusing training efforts on the more complex opening and middle game phases.

---

#### **Inference Process**

The inference strategy for the current chess AI system represents a major evolution from previous models. Unlike prior approaches, which relied on a single model trained on full games, the current system is divided into three distinct phases: opening, middle game, and endgame. Each phase is addressed by a specialized model or resource, allowing for a more tailored and effective inference process.

To determine which phase applies during gameplay, the system splits the game as follows:
- **Opening Phase**: First 10 moves of the game.
- **Endgame Phase**: Positions with seven or fewer pieces on the board.
- **Middle Game Phase**: Everything in between.

Below are the specific inference considerations for each phase:

##### **1. Opening Inference**

The opening model is lightweight, leveraging patterns from annotated grandmaster games and opening books to predict the best move during the first 10 moves of a game.

- **Process**:
  - **Inputs**:
    - Current **FEN** string representing the board position.
    - Array of up to 10 previous **UCI moves**.
  - **Outputs**:
    - **Move Probabilities**: Array of probabilities for all legal moves.
    - **Predicted Game Outcome**: Win, draw, or loss based on the highest-probability move.
  - **Steps**:
    - Use the vocabulary mappings saved during training to convert the input FEN and moves into tensors.
    - Pass the tensors into the trained opening transformer model.
    - Convert the model's output back to a UCI move using the saved vocabulary.

- **Strengths**:
  - Captures common opening patterns effectively.
  - Fast inference due to the model's simplicity.

- **Limitations**:
  - Focuses on general strategies, which may not adapt well to unorthodox openings.

##### **2. Middle Game Inference**

The middle game is significantly more complex than the opening, requiring a sophisticated inference process to balance policy outputs and position evaluations.

- **Process**:
  - **Inputs**:
    - Current **FEN** string.
    - Full sequence of prior **UCI moves** leading to the position.
  - **Outputs**:
    - Predicted **best move** using a combination of strategies:
      1. **Policy-Only**: Use the model's probability distribution directly.
      2. **Weighted Evaluation**: Combine policy probabilities with value outputs (centipawn and mate evaluations).
      3. **Alpha-Beta Pruning**: Search moves to a specified depth, pruning suboptimal branches based on weighted scores.
  - **Key Strategies**:
    1. **Policy-Only**:
       - Select the move with the highest probability directly from the model's output.
       - Observed to cause unnecessary blunders due to insufficient confidence in top moves.
    2. **Weighted Evaluation**:
       - Normalize **centipawn (CP)** evaluations to a [-1, 1] range.
       - Incorporate **mate evaluations**, penalizing negative mates (opponent's advantage) and rewarding low-positive values (proximity to winning).
       - Add penalties for repeated moves and cycles to encourage exploration.
       - Use dynamic weighting to prioritize:
         - High CP values in the early middle game.
         - King safety and check-inducing moves in the late middle game (when fewer than 14 pieces remain on the board).
    3. **Alpha-Beta Pruning**:
       - Evaluate moves at increasing depths, scoring branches based on CP and mate evaluations.
       - Prune unpromising branches to focus on high-potential moves.
       - Despite its potential, this method is computationally expensive and relies on accurate policy outputs.

- **Strengths**:
  - Adapts dynamically to strategic nuances.
  - Effective at mitigating the model's weaknesses by leveraging value outputs for better decisions.

- **Limitations**:
  - Weighted evaluations require fine-tuning to balance priorities effectively.
  - Alpha-beta pruning is computationally intensive and limited by model accuracy.

##### **3. Endgame Inference**

For positions with seven or fewer pieces, the system relies on the Lichess 7-piece tablebase to ensure perfect play.

- **Process**:
  - **Inputs**:
    - Current **FEN** string.
  - **Outputs**:
    - **Best Move** with associated win-draw-loss value.
  - **Steps**:
    - Query the Lichess tablebase API with the current FEN.
    - Retrieve a list of legal moves, each with a win-draw-loss score.
    - Select the move with the highest win probability or lowest loss probability.
  - **Fallback**:
    - If API access fails, fallback strategies prioritize conservative moves to avoid blunders.

- **Strengths**:
  - Guarantees optimal play in the endgame phase.
  - Reduces the need for training endgame-specific models.

- **Limitations**:
  - Relies on internet access unless the tablebase is stored locally (storage requirements: 17TB for 7-piece positions).
  - Limited to positions with seven or fewer pieces.

By dividing the game into these three phases and tailoring inference strategies accordingly, the current system achieves a significant improvement over previous monolithic approaches. Each phase leverages specialized resources or models to maximize accuracy and strategic depth, making the system more robust and adaptable to various gameplay scenarios.

#### Previous Models

Before arriving at the current transformer-based model, simpler models were developed and tested. Each iteration provided insights into the challenges of training a chess bot, ultimately guiding the design of the transformer architecture.

##### 1. Deep Neural Network (DNN)

The Deep Neural Network (DNN) was implemented as an initial exploration to understand the capabilities of basic neural networks in learning chess moves. This model focused on processing individual board positions and their associated moves without leveraging sequential game context.

- **Data Preparation**:
  - Chess positions (FEN) were paired with their corresponding moves (UCI format) and labeled for training.
  - FEN strings were converted into an 8x8 matrix representation, where pieces were mapped to integer values (e.g., pawns as ±1, knights as ±2).
  - Moves were flattened into a single index (0–4095), representing all possible source and destination combinations.
  - A CSV file was created with the following columns:
    - **FEN**: Chessboard position.
    - **Move**: Move in UCI format.
    - **White ELO**: Elo rating of the white player.
    - **Black ELO**: Elo rating of the black player.
  - The dataset was split into 80% training and 20% testing data.

- **Model Architecture**:
  - A Keras sequential model was constructed with the following layers:
    - **Flatten**: Processes the 8x8 matrix input.
    - **Dense (128 units)**: Extracts features using ReLU activation.
    - **Dense (64 units)**: Further refines features using ReLU activation.
    - **Output Dense (4096 units)**: Predicts the best move using softmax activation.

- **Training**:
  - The model was trained for 10 epochs using 5k games per ELO range:
    - **<1000, 1000–1500, 1500–2000, >2000.**
  - Independent models were trained for each ELO range.

- **Results**:
  Despite completing training, the DNN struggled to generalize chess rules effectively, as shown below:

  | ELO Range      | Loss   | Accuracy |
  |----------------|--------|----------|
  | **<1000**      | 5.9137 | 9.77%    |
  | **1000–1500**  | 5.7089 | 9.99%    |
  | **1500–2000**  | 5.7176 | 9.41%    |
  | **>2000**      | 5.7024 | 8.83%    |

- **Key Insights**:
  - The DNN memorized moves but lacked situational awareness, making it unable to adapt to chess rules or strategies.
  - The architecture struggled to learn chess principles, even with scaled datasets.
  - Low accuracy highlighted the need for models capable of understanding positional and sequential aspects of the game, leading to the exploration of more advanced architectures.

##### 2. Convolutional Neural Network (CNN)

To improve the bot's understanding of chess-specific attributes, a Convolutional Neural Network (CNN) was implemented. Unlike the DNN, the CNN incorporates both the board's spatial representation and additional game-specific features, such as castling rights and move clocks, to make better-informed predictions.

- **Key Improvements**:
  - Added game-specific features to the dataset:
    - Turn (white or black).
    - Castling rights (king- and queen-side for each player).
    - En passant availability.
    - Half-move and full-move clocks.
  - Represented the board spatially using an 8x8 matrix.
  - Enhanced the model's ability to interpret spatial relationships between pieces.

- **Data Preparation**:
  - **FEN Conversion**:
    - Chess positions (FEN) were converted into:
      - An 8x8 matrix, with pieces mapped as integers (e.g., pawns as ±1).
      - An additional feature vector capturing turn, castling rights, en passant, and move clocks.
    - These components were combined to form the input data.
  - **Dataset Creation**:
    - Each move was paired with its corresponding FEN and game attributes in a CSV file.
    - Moves were indexed using a flattened representation (0–4095 for all possible UCI moves).
  - **Train-Test Split**:
    - The dataset was split into 80% training and 20% testing data.

- **Model Architecture**:
  - A multi-input CNN model was designed using the Keras Functional API:
    - **Board Input**:
      - Conv2D layers processed the spatial relationships of the 8x8 board.
      - MaxPooling2D layers reduced dimensionality.
      - A Flatten layer prepared the output for dense layers.
    - **Feature Input**:
      - A separate input layer processed the additional feature vector.
    - **Combined Network**:
      - Both inputs were concatenated and passed through dense layers.
      - A final softmax layer predicted the best move out of 4096 possible moves.

- **Training and Results**:
  - The CNN was trained independently for each ELO range using 5k games per range for 30 epochs.
  - Early stopping was applied to prevent overfitting.

  | ELO Range      | Loss   | Accuracy |
  |----------------|--------|----------|
  | **<1000**      | 5.0660 | 10.77%   |
  | **1000–1500**  | 5.0810 | 10.46%   |
  | **1500–2000**  | 5.1264 | 10.24%   |
  | **>2000**      | 5.1905 | 9.66%    |

- **Insights and Limitations**:
  - **Improvements**:
    - The CNN demonstrated a slight improvement over the DNN, achieving better accuracy across all ELO ranges.
    - Incorporating chess-specific features helped the model interpret positional attributes more effectively.
  - **Challenges**:
    - The model's accuracy remained low, indicating:
      - Limited understanding of the rules of chess.
      - Lack of sequential awareness, as moves were evaluated independently without considering the flow of the game.

##### 3. Convolutional Neural Network (CNN) Unified Variant

Building on previous experiments, this model introduces a unified approach by incorporating player ELO ratings into the feature data. The goal was to train a single base model on a larger dataset from advanced (1500–2000) and master-level (>2000) players, followed by fine-tuning for specific ELO ranges.

- **Key Enhancements**:
  - **Incorporation of Player ELOs**:
    - Added the average ELO of the two players and their ELO range as additional features.
  - **Unified Model Training**:
    - Trained a single base model on a combined dataset of 15k games, simplifying the training process and increasing the volume of data for better generalization.
  - **Focus on Advanced Games**:
    - Limited training data to higher ELO ranges (1500 and above) to prioritize learning from stronger chess strategies.

- **Data Preparation**:
  - **FEN Conversion**:
    - Chessboard positions were converted into:
      - **8x8 Matrix**: Encodes piece positions as integers.
      - **Feature Vector**: Captures turn, castling rights, en passant availability, move clocks, average ELO, and ELO range.
    - Example additional features:
      - 1 for White's turn, 0 otherwise.
      - Castling rights (K, Q, k, q).
      - En passant target square (file index) or -1 if unavailable.
      - Average ELO and its range.
  - **Dataset Creation**:
    - Each move was paired with its corresponding FEN and features in a CSV file.
    - Moves were indexed using a flattened representation (0–4095 for all possible UCI moves).
  - **Train-Test Split**:
    - The dataset was split into 80% training and 20% testing data.

- **Model Architecture**:
  - The unified CNN model uses the Keras Functional API and incorporates multiple inputs:
    - **Board Input**:
      - Conv2D layers with increasing filters (64, 128, 256) to capture spatial relationships.
      - MaxPooling2D layers for dimensionality reduction.
      - Flattened output for combination with feature input.
    - **Feature Input**:
      - Dense layers process the feature vector (ELO-related and game-specific attributes).
    - **Combined Network**:
      - Concatenated board and feature inputs pass through dense layers with dropout to reduce overfitting.
      - A final softmax layer predicts the best move out of 4096 possibilities.

- **Training and Results**:
  - The base model was trained for 30 epochs on 15k games from players with ELOs of 1500 and above.
  - Early stopping and learning rate reduction were used to improve training stability.

  | Dataset         | Loss   | Accuracy |
  |-----------------|--------|----------|
  | **Base Model**  | 5.2731 | 8.95%    |

- **Insights and Limitations**:
  - **Challenges with ELO Features**:
    - Including player ELO in the feature vector did not improve performance. Instead, it introduced unnecessary complexity, hindering the model’s ability to learn chess rules effectively.
    - The additional features created noise, diluting the meaningful patterns the model needed to learn.
  - **Unified Training**:
    - Training a single base model on a larger dataset showed potential for better generalization compared to smaller, independent models.
  - **Future Directions**:
    - Focus on essential game features and refine feature engineering to improve the model's ability to predict valid and strategic moves.

##### 4. Convolutional Neural Network (CNN) Legal Moves Variant

After observing that adding more games and features led to poorer performance, this variant shifted focus to a more targeted approach: training the model exclusively on legal moves available at the current FEN. The goal was to simplify the task and help the model focus on valid decisions.

- **Key Features**:
  - **Legal Move Masking**:
    - Leveraged the `chess.js` API to generate a mask of all valid moves for each FEN.
    - This mask was used to restrict the model’s predictions to legal moves only.
  - **Enhanced Board Representation**:
    - The board was encoded as an 8x8x12 matrix, with each channel representing a specific piece type (e.g., pawns, knights).
  - **Simplified Problem Scope**:
    - Limited training data to games from players with ELOs 1500 and above (advanced and master ranges).

- **Data Preparation**:
  - **FEN Conversion**:
    - FEN strings were converted into:
      - **8x8x12 Matrix**: Encodes the spatial arrangement of pieces.
      - **Legal Moves Mask**: Restricts predictions to valid moves.
    - Example: A pawn move like e2e4 would only be considered valid if it is legal at the given FEN.
  - **Dataset Creation**:
    - Each move was paired with its FEN and legal move mask in a CSV file.
    - Moves were indexed using a flattened representation (0–4095 for all possible UCI moves).
  - **Train-Test Split**:
    - The dataset was split into 80% training and 20% testing data.

- **Model Architecture**:
  - The CNN Legal Moves Variant incorporated multiple inputs for board data and legal move masks:
    - **Board Input**:
      - Conv2D layers to extract spatial relationships.
      - MaxPooling2D layers for dimensionality reduction.
      - Flattened output passed through dense layers.
    - **Legal Moves Mask Input**:
      - Multiplied with the model’s raw logits to enforce legal move predictions.
    - **Combined Network**:
      - The model’s output was restricted to legal moves using a softmax layer applied over the masked logits.

- **Training and Results**:
  - The model was trained for 30 epochs on 15k games from advanced and master players.

  | Dataset         | Loss   | Accuracy |
  |-----------------|--------|----------|
  | **Base Model**  | 7.9835 | 3.47%    |

- **Insights and Limitations**:
  - **Improvements**:
    - By focusing only on legal moves, the model avoided nonsensical predictions, ensuring all outputs adhered to chess rules.
  - **Challenges**:
    - The model’s accuracy remained very low (3.47%), indicating difficulty in learning optimal moves even when restricted to legal options.
    - The sequential and strategic nature of chess still posed significant challenges, as the CNN evaluated moves independently, without understanding the broader game context.
  - **Key Takeaway**:
    - The inability of the CNN to grasp the sequential complexities of chess led to the exploration of transformer-based models. Transformers, with their ability to analyze entire game sequences, were identified as a more promising architecture.
  
##### 5. Sequence-Based Transformer

To address the lack of sequential understanding in CNN models, a transformer-based architecture was introduced. This model leverages the sequential nature of chess games to predict moves based on the flow of play.

- **Key Features**:
  - **SAN Move Vocabulary**:
    - Assigned a unique index to every chess move in the dataset, mapped from the SAN (Standard Algebraic Notation) format.
  - **Tokenized Games**:
    - Converted each game's sequence of moves from SAN to index values using the vocabulary.
  - **Evaluation Metrics**:
    - Assessed the model's success by removing the last move from a game's sequence and predicting all moves, including the missing one.

- **Data Preparation**:
  - **PGN Chess Games**:
    - Extracted games and categorized them into four ELO ranges:
      - **Novice** (<1000)
      - **Intermediate** (1000–1500)
      - **Advanced** (1500–2000)
      - **Master** (>2000)
  - **Move Vocabulary Creation**:
    - Created a vocabulary JSON file, assigning unique IDs to every move in the dataset.
  - **Tokenization**:
    - All moves were tokenized by mapping them to their respective vocabulary IDs.
  - **Dataset Splitting**:
    - Split the dataset into 80% training and 20% testing sets.

- **Model Architecture**:
  - A Keras transformer model was built with:
    - **Multi-head Attention**:
      - Captures relationships between moves.
    - **Dropout**:
      - Prevents overfitting during training.
    - **Layer Normalization**:
      - Stabilizes training for improved convergence.
    - **Dense Layers**:
      - Processes features for move predictions.

- **Training and Results**:
  - The model was trained on 15k and 50k game datasets for 10 epochs each.
  - Results with the 50k dataset:

  | Dataset         | Loss   | Accuracy |
  |-----------------|--------|----------|
  | **Base Model**  | 0.5491 | 89.10%   |

- **Insights and Limitations**:
  - **Improvements**:
    - Accurately predicted the sequence of moves for most games, demonstrating an improved understanding of chess flow compared to CNN models.
  - **Challenges**:
    - Despite high accuracy (~90%), inference probabilities were low (~1–2% for the best move).
    - The sequence-based approach does not align with real-time inference needs, where only the current game state is available.
  - **Key Takeaway**:
    - The transformer-based model excelled in analyzing sequential patterns and demonstrated a beginner-level understanding of chess, equivalent to an ELO below 1000.

##### 6. Fine-Tuned Models for ELO Ranges

Building on the base transformer model, fine-tuning was performed to adapt the model for specific ELO ranges, ensuring tailored gameplay for players of different skill levels.

- **Key Features**:
  - Leveraged the base transformer model as a foundation, refining it for each ELO range (<1000, 1000–1500, 1500–2000, >2000).
  - Maintained the original move vocabulary and tokenized datasets for consistency.

- **Data Preparation**:
  - **ELO Categorization**:
    - Extracted new PGN games and categorized them into the same ELO ranges as above.
  - **Move Tokenization**:
    - Tokenized moves using the vocabulary created during the base model's training.
  - **Dataset Splitting**:
    - Split the dataset into 80% training and 20% testing.

- **Fine-Tuning Process**:
  - Loaded the pretrained base model.
  - Fine-tuned for 5 epochs using 15k games per ELO range.

- **Training and Results**:
  - The fine-tuned models achieved the following results:

  | ELO Range      | Loss   | Accuracy |
  |----------------|--------|----------|
  | **<1000**      | 0.4100 | 91.81%   |
  | **1000–1500**  | 0.4782 | 90.36%   |
  | **1500–2000**  | 0.5559 | 88.88%   |
  | **>2000**      | 0.6249 | 87.61%   |

- **Insights and Limitations**:
  - **Improvements**:
    - Fine-tuning allowed the model to simulate skill-appropriate gameplay for each ELO range.
    - Achieved higher accuracy and lower loss compared to earlier architectures. 
  - **Challenges**:
    - While predictions were more accurate, some inferred moves still exhibited low probabilities during inference.
  - **Key Takeaway**:
    - Fine-tuning successfully tailored the model to different skill levels, improving gameplay realism and strategy. However, real-world usage may require further adjustments to inference techniques, such as training the model on partial games where the next move is unknown.

##### 7. Game Phase Transformer

The Game Phase Transformer aimed to enhance chess move predictions by segmenting games into phases (opening, mid-game, and end-game) and training the model to handle phase-specific strategies. By incorporating dynamic partial game generation and board features, this approach sought to improve accuracy across different stages of the game.

- **Key Features**:
  - **ELO-Based Game Filtering**:
    - Filters chess games from a PGN file based on specified ELO ranges, ensuring skill-specific datasets for training and evaluation.
  - **Move Vocabulary Creation**:
    - Builds a vocabulary of chess moves using SAN (Standard Algebraic Notation), assigning unique indices to each move for tokenization.
  - **Game Phase Recognition**:
    - Dynamically categorizes moves into game phases (opening, mid-game, end-game) based on board state and move count.
  - **Dynamic Partial Game Generation**:
    - Generates partial game sequences dynamically for training, providing additional variability and emphasizing move prediction at different stages.
  - **Transformer Model Integration**:
    - Implements a transformer-based architecture incorporating multi-head attention, board features, and game phases to predict the next move in chess games.
  - **Performance Visualization**:
    - Includes plotting capabilities to visualize training and validation loss and accuracy trends across epochs.

- **Data Preparation**:
  - **Game Filtering by ELO Range**:
    - Chess games are filtered based on ELO ranges (e.g., <1000, 1000–1500, etc.).
    - Games are extracted and validated for a minimum number of moves to ensure meaningful training samples.
  - **Move Vocabulary Construction**:
    - A vocabulary is created by mapping all unique SAN moves across the dataset to integer indices, stored as JSON for reuse.
  - **Tokenization**:
    - Games are tokenized into sequences of move indices using the vocabulary, with each sequence representing a game.
  - **Feature Extraction**:
    - Board features are represented as 8x8x12 tensors, encoding piece positions and types for each square.
    - Game phases are labeled dynamically as "opening," "mid-game," or "end-game" based on move count and board complexity.
  - **Dataset Splitting**:
    - The dataset is split into 80% training and 20% testing for full games.
    - Partial game data is generated dynamically during training.

- **Model Architecture**:
  - **Inputs**:
    - **Move Sequences**: Tokenized sequences of moves from partial or full games.
    - **Board Features**: Tensor representations of the chessboard at each move.
    - **Game Phases**: Encoded as categorical embeddings for "opening," "mid-game," and "end-game."
  - **Embedding and Attention Layers**:
    - Move sequences are embedded and processed through multi-head attention layers to capture relationships between moves.
    - Dropout and layer normalization are applied to stabilize training and prevent overfitting.
  - **Board and Phase Integration**:
    - Dense layers process board features and game phase embeddings.
    - Features are concatenated with move embeddings to enhance the model’s contextual understanding.
  - **Prediction**:
    - The model outputs logits for all possible moves, focusing on the final move in a sequence for next-move prediction.
  - **Model Output**:
    - Predicts the next move in a chess game, given the partial game sequence, current board state, and game phase.

- **Training and Results**:
  - The model was trained on 15k games for 10 epochs.
  - Achieved the following results with the 15k dataset:

  | Dataset         | Loss   | Sparse Categorical Accuracy |
  |-----------------|--------|-----------------------------|
  | **Base Model**  | 8.3499 |             0.41%           |

- **Insights and Limitations**:
  - **Improvements**:
    - **Game Phase Awareness**: Including game phases as features helped the model differentiate between early, middle, and endgame strategies.
    - **Board State Integration**: Incorporating board features improved the model’s contextual understanding of positions.
    - **Transformer Advantages**: Multi-head attention effectively captured sequential dependencies between moves, outperforming simpler architectures.
  - **Challenges**:
    - **Low Accuracy**: Despite capturing game context, the model struggled to achieve meaningful accuracy due to the vast move space and complex positional reasoning in chess.
    - **Inference Complexity**: Predicting moves based on full game sequences may not align perfectly with real-time gameplay scenarios.
  - **Key Takeaway**:
    - The transformer model showed promise in handling chess move sequences, but further refinement is needed to improve accuracy and adapt to real-world use cases.
    - Future work could include integrating domain-specific heuristics (e.g., Stockfish evaluations) or fine-tuning the model for specific gameplay scenarios.

##### 8. **ELO-Balanced Transformer**:

The ELO-Balanced Transformer aimed to address dataset imbalance by ensuring equitable representation of games across various skill levels. This approach enhanced training diversity while leveraging the advantages of transformer-based architectures for sequence prediction.

- **Key Features**:
  - ELO-Based Dataset Balancing**:
      - Filters chess games by ELO ranges and applies weighted sampling to ensure balanced representation across skill levels.
  - **Move Vocabulary Creation**:
      - Builds a vocabulary of chess moves in SAN (Standard Algebraic Notation), mapped to unique indices for tokenization.
  - **Next-Move Prediction**:
      - Prepares training datasets to predict the next move in a game based on partial move sequences.
  - **Transformer Model Integration**:
      - Implements a transformer-based architecture with multi-head attention to capture sequential dependencies.
  - **Performance Tracking and Visualization**:
      - Includes metrics and plotting capabilities to visualize training and validation trends.
  - **Model Saving and Reusability**:
      - Saves trained models and vocabulary mappings for future inference or fine-tuning.
- **Data Preparation**:
  - **Game Filtering by ELO Range**:
      - Chess games are filtered based on predefined ELO ranges: `<1000`, `1000–1500`, `1500–2000`, and `2000+`.
      - Only games with sufficient moves are included in the dataset.
  - **Weighted Sampling**:
      - Applies inverse weighting to underrepresented ELO ranges, ensuring equitable contributions to the training process.
  - **Move Vocabulary Construction**:
      - A vocabulary is created by mapping all unique SAN moves to integer indices, stored in JSON for reuse.
  - **Tokenization**:
      - Games are tokenized into sequences of move IDs, representing partial or full games.
  - **Dataset Splitting**:
      - The tokenized data is split into 80% training and 20% validation sets.
- **Model Architecture**:
  - **Inputs**:
      - **Move Sequences**: Tokenized partial game sequences, padded to the maximum sequence length.
  - **Embedding and Attention Layers**:
      - An embedding layer maps moves to dense vectors.
      - Multi-head attention layers process sequential dependencies between moves.
  - **Feedforward Layers**:
      - Fully connected layers process the output of attention layers, with dropout and normalization for stability.
  - **Prediction Output**:
      - A softmax layer predicts the next move based on the partial sequence, focusing on the final move in the input sequence.
  - **Model Output**:
      - Probabilities for all possible moves, where the highest-probability move is selected as the prediction.

- **Training and Results**:
  - The model was trained on 15k games for 10 epochs.
  - Achieved the following results with the 15k dataset:

    | Dataset         | Loss   | Sparse Categorical Accuracy |
    |-----------------|--------|-----------------------------|
    | **Base Model**  | 5.6630 |             4.62%           |

- **Insights and Limitations**:
  - **Improvements**:
      - **Balanced Dataset**: Weighted sampling effectively addressed the overrepresentation of common ELO ranges, ensuring more equitable training data.
      - **Transformer Advantages**: Multi-head attention captured sequential dependencies better than simpler architectures like CNNs.
  - **Challenges**:
      - **Low Accuracy**: Despite addressing dataset imbalance, the model’s accuracy (~4.6%) indicates difficulty in generalizing chess strategies from partial sequences.
      - **Loss Plateau**: The model struggled to reduce validation loss, suggesting potential overfitting or insufficient feature complexity.
  - **Key Takeaways**:
      - Dataset balancing is a step forward, but move prediction remains challenging due to the large move space and context-dependent nature of chess.
      - Future improvements could include integrating domain knowledge (e.g., legal move masks or Stockfish evaluations) or experimenting with more sophisticated architectures.

##### 9. **Policy and Value Transformer**:

The Policy and Value Transformer introduced a dual-output architecture to simultaneously predict the next move (policy) and evaluate the current board position (value). This approach enabled the model to combine strategic move selection with positional analysis.

- **Key Features**:
  - **Chessboard Encoding**:
      - Converts chessboard states from FEN into an 8x8x16 tensor representation.
      - Encodes pieces, turn, castling rights, and en passant square into separate channels.
  - **Move Vocabulary Creation**:
      - Generates a mapping of all possible UCI moves (including promotions) to unique indices.
  - **Training Data Preparation**:
      - Processes JSONL files to extract FEN, previous moves, next move, and Stockfish evaluations (CP and mate values).
      - Includes move history (up to 5 previous moves) for better context.
  - **Dual-Output Model**:
      - Implements a Transformer architecture with separate outputs for policy (next-move prediction) and value (board evaluation).
  - **Loss and Metrics**:
      - Tracks combined policy and value loss, with specific metrics for cross-entropy (policy) and MSE (value).
      - Includes top-3 accuracy for policy predictions and MAE for value predictions.
  
- **Data Preparation**:
  - **Input Features**:
      - Board states are encoded into tensors representing piece positions, turn, castling rights, and en passant square.
      - Move history is tokenized and padded to a fixed length of 5 moves.
  - **Labels**:
      - Policy: The next move is encoded using the move vocabulary.
      - Value: Stockfish evaluations are normalized to [0, 1] for regression tasks.
  - **Dataset Splitting**:
      - The dataset is split into 80% training and 20% validation sets.
  - **Batching**:
      - Data is provided in batches using a generator, combining board states, move histories, policy labels, and value labels.

- **Model Architecture**:
  - **Inputs**:
      - **Board Input**: Encoded board state tensors (8x8x16).
      - **History Input**: Tokenized and padded move histories.
  - **Transformer Layers**:
      - Combines board and move history features using multi-head attention layers.
      - Applies feedforward layers with residual connections for further feature extraction.
  - **Output Heads**:
      - **Policy Head**: Predicts the probability distribution of all possible moves.
      - **Value Head**: Predicts the board evaluation as a regression task.

- **Training and Results**:
  - The model was trained with the following results across checkpoints:

    | Checkpoint | Loss   | Policy Loss | Value Loss | Policy Accuracy | Top-3 Accuracy | Value MAE |
    |------------|--------|-------------|------------|-----------------|----------------|-----------|
    | **2**      | 5.6935 | 5.6935      | 1.2366e-05 | 4.21%           | 10.25%         | 0.0023    |
    | **3**      | 5.5582 | 5.5582      | 1.2631e-05 | 4.74%           | 11.38%         | 0.0025    |
    | **4**      | 5.6164 | 5.6163      | 6.2147e-04 | 4.48%           | 10.81%         | 0.0187    |

- **Insights and Limitations**:
  - **Improvements**:
      - **Dual-Output Model**: Simultaneously predicts the next move (policy) and board evaluation (value), leveraging a shared feature representation.
      - **Move History**: Including move history improved context for next-move prediction, capturing sequential patterns.
      - **Top-3 Accuracy**: The model demonstrated better performance in identifying likely moves within the top 3 predictions, suggesting it captures partial positional understanding.
  - **Challenges**:
      - **Handling CP and Mate Evaluations**: Combining centipawn (CP) and mate evaluations into a single value proved ineffective, as it conflates very different aspects of positional evaluation. This limited the model's ability to learn nuanced board assessments.
      - **Legal Move Context**: The absence of legal move evaluations (e.g., Stockfish scores for all legal moves) restricted the model's understanding of tactical and positional trade-offs.
      - **Dominance of Frequent Moves**: Common moves across different phases (e.g., opening or endgame) dominated predictions, reducing accuracy for less frequent, phase-specific strategies.
  - **Key Takeaways**:
      - **Phase-Specific Models**: Training individual models for opening, middle game, and endgame phases could help prevent dominant moves from overshadowing less frequent but critical ones, leading to phase-optimized strategies.
      - **Legal Move Evaluations**: Incorporating legal moves and their evaluations (e.g., centipawn/mate scores for all possible actions) would provide the model with richer tactical and positional context.
      - **Improved Value Handling**: Separating centipawn evaluations and mate scores into distinct outputs would allow the model to better differentiate between material imbalances and forced mate scenarios.

### Performance Summary

See the performance metrics of each model, highlighting their accuracy and loss across different datasets and training configurations.

Loss and accuracy are metrics used to evaluate the performance of machine learning models:

- **Loss**: Represents how far the model's predictions are from the actual outcomes. Lower loss values indicate better performance during training and testing.
- **Accuracy**: Measures the percentage of correct predictions made by the model. Higher accuracy reflects better alignment with the expected results.

| **Model**                      | **Dataset ELO** | **Games Trained** | **Loss** | **Accuracy**  |
|--------------------------------|-----------------|-------------------|----------|---------------|
| **DNN**                        | <1000           | 5k               | 5.9137   | 9.77%         |
|                                | 1000–1500       | 5k               | 5.7089   | 9.99%         |
|                                | 1500–2000       | 5k               | 5.7176   | 9.41%         |
|                                | >2000           | 5k               | 5.7024   | 8.83%         |
| **CNN**                        | <1000           | 5k               | 5.0660   | 10.77%        |
|                                | 1000–1500       | 5k               | 5.0810   | 10.46%        |
|                                | 1500–2000       | 5k               | 5.1264   | 10.24%        |
|                                | >2000           | 5k               | 5.1905   | 9.66%         |
| **CNN Unified**                | >1500           | 15k              | 5.2731   | 8.95%         |
| **CNN Legal Moves**            | >1500           | 15k              | 7.9835   | 3.47%         |
| **Base Transformer**           | All             | 15k              | 0.8449   | 84.69%        |
|                                | All             | 50k              | 0.5491   | 89.10%        |
| **Fine-Tuned Transformer**     | <1000           | 15k              | 0.4100   | 91.81%        |
|                                | 1000–1500       | 15k              | 0.4782   | 90.36%        |
|                                | 1500–2000       | 15k              | 0.5559   | 88.88%        |
|                                | >2000           | 15k              | 0.6249   | 87.61%        |
| **Game Phase Transformer**     | All             | 15k              | 8.3499   | 0.41%         |
| **ELO-Balanced Transformer**   | All             | 15k              | 5.6630   | 4.62%         |
| **Policy and Value Transformer** | All           | 15k              | 5.6164   | 4.48% (Policy) |
|                                |                 |                   |          | 10.81% (Top-3) |
| **Opening Phase Transformer**  | All             | 15k              | 2.5144   | 46.98%        |
| **Midgame Phase Transformer**  | All             | 3.2k             | 4.1148   | 15.83%        |
|                                |                 |                   |          | 38.17% (Top-5)|


### Challenges & Learnings

The multi-phase transformer model has undergone significant development, revealing key challenges and areas for improvement:

1. ELO Performance Evaluation
  - V1 Model:
    - When tested against Stockfish's lowest level (1350 ELO), the V1 model demonstrated beginner-level play (~800 ELO) as per post-game Stockfish analysis.
  - V5/V6 Model:
    - Games against Stockfish's lowest level were far more competitive, showcasing intermediate-level play (~1200 ELO).
  - Direct Comparison:
    - Simulating a match between V1 and V5/V6 confirmed these ELO estimates. The V5/V6 model consistently outperformed V1, making fewer blunders and exhibiting more strategic gameplay.
2. Challenges in Achieving Advanced and Master-Level Play
  - Data Limitations:
    - The V5/V6 model was trained on only 3,200 annotated grandmaster games due to preprocessing constraints. Consequently, the model predicted the true grandmaster move within its top 5 suggestions only 38% of the time, leading to intermediate-level play (~1200 ELO).
  - Mid-to-Endgame Transition:
    - Post-game analysis highlighted poor performance during the transition from midgame to endgame. This can be attributed to the limited dataset and the increasing complexity of possible moves in these phases.
  - Proposed Solution:
    - Expanding the dataset with more preprocessed grandmaster games is critical. Increasing the training data size could enhance the model's strategic understanding, potentially improving its ELO to the advanced (~1750) range.
3. Late-Game Puzzle Integration
  - Targeting Weak Points:
    - The mid-to-endgame transition remains the model's weakest phase. Integrating Lichess puzzles specifically designed for late-game scenarios could significantly improve performance.
  - Proposed Strategy:
    - Train a separate model exclusively on puzzles and use it to fine-tune the V5/V6 transformer. This would enhance strategic depth without losing the strengths of the base model, potentially achieving master-level play (~2250 ELO).
4. Exploring Unsupervised Learning
  - Supervised Learning Limits:
    - As more data is added, the transformer model may approach the upper limits of what supervised learning can achieve.
  - Next Steps:
    - Leverage the knowledge from the supervised model to train a fine-tuned unsupervised model.
    - Use reinforcement learning techniques to reward or penalize moves based on game outcomes, enabling the model to improve by playing against itself or Stockfish.

#### Current Progress and Future Goals
  - Current Achievements:
    - The V1 transformer model reliably plays at a beginner level (~800 ELO).
    - The V5/V6 transformer models have achieved intermediate-level play (~1200 ELO).
  - Next Milestones:
    - Develop models capable of advanced (~1750 ELO) and master (~2250 ELO) play.
    - Until these milestones are reached, Stockfish will remain the backend engine for advanced and master-level gameplay.

## Future Features

The following enhancements are planned to advance the capabilities of the chess bot and refine its gameplay:

1. Expanded Training Dataset:
  - Preprocess Stockfish FEN evaluations for all moves across 26,000 grandmaster games.
  - Evaluate every move and corresponding legal moves at each position to provide richer data for training, enhancing the model’s understanding of advanced strategies.
2. Lichess Puzzles Integration:
  - Fine-tune the model using targeted Lichess puzzles to address specific weaknesses, particularly in the mid-to-endgame transition.
  - Utilize puzzle-based training to improve the model's ability to handle complex positional and tactical scenarios.
3. Unsupervised Learning:
  - Extend the transformer model with unsupervised learning techniques to improve decision-making:
  - Train the model by playing games against itself or Stockfish.
  - Reward or penalize moves based on their effectiveness, enabling the model to refine its strategies dynamically.
4. Advanced and Master Level Bots:
  - Develop and deploy bots capable of playing at advanced (~1750 ELO) and master (~2250 ELO) levels.
  - Incorporate larger datasets, late-game puzzle strategies, and fine-tuned unsupervised learning to achieve the required performance.

These planned features will pave the way for a more robust and competitive chess bot, capable of catering to a wide range of skill levels.

## Contributing

If you want to contribute to this project, feel free to open an issue or submit a pull request. Any contributions, from bug fixes to new features, are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
