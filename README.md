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
  - [Installation](#installation)
  - [Dataset](#dataset)
- [Chess Bot Creation](#chess-bot-creation)
  - [Model Architecture](#model-architecture)
    - [Current Model](#current-model)
    - [Previous Models](#previous-models)
- [Performance Summary](#performance-summary)
- [Challenges & Learnings](#challenges--learnings)
- [Future Features](#future-features)
- [Contributing](#contributing)
- [License](#license)

## Overview

**Checkmate** is a real-time chess platform where players can compete against live opponents, chat in-game, track their rankings, and practice with human-like AI bots. The AI bot is built using TensorFlow in Python, processing games to train and test models. The trained models are exported in ONNX format for seamless integration into the Checkmate backend.

## Checkmate Demo

The Checkmate application is live and can be accessed here: [Checkmate Demo](https://checkmateplay.com). Explore all features, including human vs. human gameplay, human vs. bot gameplay, in-game chat, and player rankings.

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

## Getting Started

Follow the instructions below to set up the project on your local machine.

### Prerequisites

Ensure the following are installed:

- **Python** (v3.6 or higher)
- **VS Code** (or any preferred IDE)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AdenWhitworth/checkmate-AI.git
   ```
2. Create and activate a virtual environment
    - On windows:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```
    - On macOS/Linux:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset

1. **Download Chess Games**: Obtain publicly available chess games in PGN format from the [Linchess Database](https://database.lichess.org/#standard_games).
2. **Preprocess the Data**: Use the provided scripts in the repository to preprocess the downloaded ZST files into a format suitable for training.

## Chess Bot Creation

Discover the step-by-step process behind the development of the Checkmate AI bot, from data preparation to advanced model architectures.

### Model Architecture

The bot's design evolved through iterative improvements, starting with simple models and gradually increasing in complexity:

1. **Deep Neural Network (DNN)**:
- Served as the baseline model to understand fundamental chess patterns and evaluate the feasibility of training on the dataset.
2. **Convolutional Neural Network (CNN)**:
- Introduced spatial reasoning to recognize patterns on the chessboard.
- Unified input representation for improved scalability.
3. **CNN Unified**: 
- unified input representation to include player elos
4. **CNN with Legal Moves**:
- Enhanced prediction accuracy by filtering moves to only valid chess actions.
- Incorporated chess-specific logic into the training process.
5. **Base Transformer Model**:
- Applied a transformer architecture to improve sequence understanding and decision-making across multiple moves.
6. **Fine-Tuned Transformer**:
- Customized the transformer model for specific ELO ranges (<1000, 1000–1500, etc.), ensuring tailored gameplay for different skill levels.

#### Current Model

The Checkmate AI bot uses a transformer-based model trained on chess games across multiple ELO ranges. This model simulates human-like gameplay and adapts to different skill levels.

##### Base Model

A base model was trained using data from all ELO ranges, created with a transformer machine learning algorithm. Here’s an overview of the process:

- **Data Preparation**:
    - PGN chess games were extracted and categorized into four ELO ranges:
        - Novice (<1000)
        - Intermediate (1000–1500)
        - Advanced (1500–2000)
        - Master (>2000)
    - A vocabulary JSON file was created, assigning unique IDs to every move in the dataset.
    - All moves were tokenized by mapping them to their respective vocabulary IDs.
    - The dataset was split into training (80%) and testing (20%) sets.
- **Model Training**:
    - A Keras transformer model was built using:
        - Multi-head attention: For understanding relationships between moves.
        - Dropout: To prevent overfitting.
        - Layer normalization: For stabilizing training.
        - Dense layers: For move predictions.
    - The model was trained/tested over 10 epochs using:
        - 15k games dataset.
        - 50k games dataset.
- **Performance**:
    - The base model trained on the 50k games dataset achieved:

    | Dataset         | Loss   | Accuracy |
    |-----------------|--------|----------|
    | **Base Model**  | 0.5491 | 89.10%   |

##### Fine-Tuned Models for ELO Ranges

To simulate skill-specific gameplay, the base transformer model was fine-tuned for each ELO range:

- **Data Preparation**:
    - New PGN games were extracted and categorized into the same ELO ranges as above.
    - Moves were tokenized using the vocabulary created during the base model's training.
    - The dataset was split (80% training, 20% testing).
- **Fine-Tuning Process**:
    - The previously trained base model was loaded and fine-tuned for 5 epochs using 15k games from each ELO range.
- **Performance**:
    - The fine-tuned models achieved the following results:

    | ELO Range      | Loss   | Accuracy |
    |----------------|--------|----------|
    | **<1000**      | 0.4100 | 91.81%   |
    | **1000–1500**  | 0.4782 | 90.36%   |
    | **1500–2000**  | 0.5559 | 88.88%   |
    | **>2000**      | 0.6249 | 87.61%   |

<img width="600" src="https://github.com/AdenWhitworth/checkmate-AI/raw/main/Visuals/elo_ranges_accuracy_loss_plots/TFMR_ELOS_LOSS.png" alt="Transformer Loss Chart">
<img width="600" src="https://github.com/AdenWhitworth/checkmate-AI/raw/main/Visuals/elo_ranges_accuracy_loss_plots/TFMR_ELOS_ACCURACY.png" alt="Transformer Accuracy Chart">

- **Training Time**:
    - Training all models (base and fine-tuned) took approximately 24–48 hours on local hardware, depending on the CPU/GPU setup.

#### Previous Models

Before arriving at the current transformer-based model, simpler models were developed and tested. Each iteration provided insights into the challenges of training a chess bot, ultimately guiding the design of the transformer architecture.

1. **Deep Neural Network (DNN)**

A DNN model was implemented as the initial approach to understand how basic neural networks perform in learning chess moves.

- **Data Preparation**:
    - Each chess position (FEN) was paired with the corresponding move (UCI format) and labeled for training.
    - FEN strings were converted into an 8x8 matrix representation where pieces were mapped to integer values (e.g., pawns as ±1, knights as ±2).
    - Moves were flattened into a single index (0–4095) to represent all possible source and destination combinations.
    - A CSV file was generated with columns for:
        - FEN (chessboard position),
        - Move (in UCI format),
        - White ELO, and
        - Black ELO.
    - The dataset was split into 80% training and 20% testing data.
- **Model Architecture**:
    - A Keras sequential model was built with:
        - A Flatten layer to process the 8x8 matrix.
        - Two Dense layers for feature extraction (128 and 64 units with ReLU activation).
        - An output Dense layer with 4096 units and softmax activation to predict the best move.
- **Training**:
    - The model was trained for 10 epochs using 5k games for each ELO range:
        - <1000, 1000–1500, 1500–2000, >2000.
    - Models were trained independently for each ELO range.
- **Results**:
    - Despite training, the model struggled to generalize the rules of chess:

    | ELO Range      | Loss   | Accuracy |
    |----------------|--------|----------|
    | **<1000**      | 5.9137 | 9.77%    |
    | **1000–1500**  | 5.7089 | 9.99%    |
    | **1500–2000**  | 5.7176 | 9.41%    |
    | **>2000**      | 5.7024 | 8.83%    |

  <img width="600" src="https://github.com/AdenWhitworth/checkmate-AI/raw/main/Visuals/elo_ranges_accuracy_loss_plots/DNN_ELOS_LOSS.png" alt="DNN Loss Chart">
  <img width="600" src="https://github.com/AdenWhitworth/checkmate-AI/raw/main/Visuals/elo_ranges_accuracy_loss_plots/DNN_ELOS_ACCURACY.png" alt="DNN Loss Chart">
  
- **Key Insights**:
    - The DNN memorized moves but lacked the situational awareness required to play chess effectively.
    - It struggled to learn the rules of chess, even with scaled datasets.
    - The low accuracy highlighted the need for a model architecture that could better understand positional and sequential aspects of the game.

2. **Convolutional Neural Network (CNN)**

To improve the bot's awareness of chess-specific attributes, a Convolutional Neural Network (CNN) was implemented. Unlike the DNN, the CNN incorporates both the board's spatial representation and additional game-specific features, such as castling rights and move clocks, to make better-informed predictions.

- **Key Improvements**:
    - Added game-specific features to the dataset:
        - Turn (white or black).
        - Castling rights (king- and queen-side for each player).
        - En passant availability.
        - Half-move and full-move clocks.
    - Used a spatial representation of the board via an 8x8 matrix.
    - Enhanced the model's ability to interpret spatial relationships between pieces.
- **Data Preparation**:
    - FEN Conversion:
        - Chess positions (FEN) were converted into:
            - An 8x8 matrix, with pieces mapped as integers (e.g., pawns as ±1).
            - An additional feature vector capturing turn, castling rights, en passant, and move clocks.
        - Both components were combined to form the input data.
    - Dataset Creation:
        - Each move was paired with the corresponding FEN and game attributes in a CSV file.
        - Moves were indexed using a flattened representation (0–4095 for all possible UCI moves).
    - Train-Test Split:
        - The dataset was split into 80% training and 20% testing data.
- **Model Architecture**:
    - A multi-input CNN model was designed using Keras Functional API:
        - Board Input:
            - Conv2D layers to process the spatial relationships of the 8x8 board.
            - MaxPooling2D layers for dimensionality reduction.
            - Flatten layer to prepare for dense layers.
        - Feature Input:
            - A separate input layer to process the additional feature vector.
        - Combined Network:
            - Both inputs were concatenated and passed through dense layers.
            - A final softmax layer predicted the best move out of 4096 possible moves.
- **Training and Results**:
    - The CNN was trained independently for each ELO range using 5k games per range, for 30 epochs. Early stopping was used to prevent overfitting.
  
    | ELO Range      | Loss   | Accuracy |
    |----------------|--------|----------|
    | **<1000**      | 5.0660 | 10.77%   |
    | **1000–1500**  | 5.0810 | 10.46%   |
    | **1500–2000**  | 5.1264 | 10.24%   |
    | **>2000**      | 5.1905 | 9.66%    |

  <img width="600" src="https://github.com/AdenWhitworth/checkmate-AI/raw/main/Visuals/elo_ranges_accuracy_loss_plots/CNN_ELOS_LOSS.png" alt="CNN Loss Chart">
  <img width="600" src="https://github.com/AdenWhitworth/checkmate-AI/raw/main/Visuals/elo_ranges_accuracy_loss_plots/CNN_ELOS_ACCURACY.png" alt="CNN Loss Chart">

- **Insights and Limitations**:
    - The CNN demonstrated a slight improvement over the DNN, with better accuracy across all ELO ranges.
    - Incorporating chess-specific features helped the model interpret positional attributes better.
    - However, the model's accuracy remained low, indicating:
        - Limited understanding of the rules of chess.
        - Lack of sequential awareness, as moves were still evaluated independently without considering the game's flow. 

3. **Convolutional Neural Network (CNN) Unified Variant**

Building on previous experiments, this model introduces a unified approach that incorporates player ELO ratings into the additional feature data. The aim was to train a single base model on a larger dataset from advanced (1500–2000) and master-level (>2000) players, followed by fine-tuning for specific ELO ranges.

- **Key Enhancements**:
    - Incorporation of Player ELOs: Added the average ELO of the two players and their ELO range as additional features.
    - Unified Model Training: A single base model was trained on a combined dataset of 15k games, simplifying the training process and increasing the volume of data for better generalization.
    - Focus on Advanced Games: Limited training data to higher ELO ranges (1500 and above) to prioritize learning from stronger chess strategies.
- **Data Preparation**:
    - FEN Conversion:
        - Chessboard positions were converted into:
            - 8x8 matrix: Encodes piece positions as integers.
            - Feature vector: Captures turn, castling rights, en passant availability, move clocks, average ELO, and ELO range.
        - Example additional features:
            - 1 for White's turn, 0 otherwise.
            - Castling rights (K, Q, k, q).
            - En passant target square (file index) or -1 if unavailable.
            - Average ELO and its range.
    - Dataset Creation:
        - Each move was paired with its corresponding FEN and features in a CSV file.
        - Moves were indexed using a flattened representation (0–4095 for all possible UCI moves).
    - Train-Test Split:
        - The dataset was split into 80% training and 20% testing data.
- **Model Architecture**:
    - The unified CNN model uses the Keras Functional API and incorporates multiple inputs:
        - Board Input:
            - Conv2D layers with increasing filters (64, 128, 256) to capture spatial relationships.
            - MaxPooling2D layers for dimensionality reduction.
            - Flattened output for combination with feature input.
        - Feature Input:
            - Dense layers process the feature vector (ELO-related and game-specific attributes).
        - Combined Network:
            - Concatenated board and feature inputs pass through dense layers with dropout to reduce overfitting.
            - A final softmax layer predicts the best move out of 4096 possibilities.
- **Training and Results**:
    - The base model was trained for 30 epochs on 15k games from players with ELOs of 1500 and above. Early stopping and learning rate reduction were used to improve training stability.
      
    | Dataset         | Loss   | Accuracy |
    |-----------------|--------|----------|
    | **Base Model**  | 5.2731 | 8.95%    |
  
- **Insights and Limitations**:
    - Challenges with ELO Features:
        - Including player ELO in the feature vector did not improve performance. Instead, it introduced unnecessary complexity, hindering the model’s ability to learn chess rules effectively.
        - The additional features created noise, diluting the meaningful patterns the model needed to learn.
    - Unified Training:
        - Training a single base model on a larger dataset showed potential for better generalization compared to smaller, independent models.
    - Future Directions:
        - Focus on essential game features and refine feature engineering to improve the model's ability to predict valid and strategic moves.

4. **Convolutional Neural Network (CNN) Legal Moves Variant**

After observing that adding more games and features led to poorer performance, this variant shifted focus to a more targeted approach: training the model exclusively on legal moves available at the current FEN. The goal was to simplify the task and help the model focus on valid decisions.

- **Key Features**:
    - Legal Move Masking:
        - Leveraged the chess.js API to generate a mask of all valid moves for each FEN.
        - This mask was used to restrict the model’s predictions to legal moves only.
    - Enhanced Board Representation:
        - The board was encoded as an 8x8x12 matrix, with each channel representing a specific piece type (e.g., pawns, knights).
    - Simplified Problem Scope:
    - Limited training data to games from players with ELOs 1500 and above (advanced and master ranges).
- **Data Preparation**:
    - FEN Conversion:
        - FEN strings were converted into:
            - 8x8x12 matrix: Encodes the spatial arrangement of pieces.
            - Legal moves mask: Restricts predictions to valid moves.
        - Example: A pawn move like e2e4 would only be considered valid if it is legal at the given FEN.
    - Dataset Creation:
        - Each move was paired with its FEN and legal move mask in a CSV file.
        - Moves were indexed using a flattened representation (0–4095 for all possible UCI moves).
    - Train-Test Split:
        - The dataset was split into 80% training and 20% testing data.
- **Model Architecture**:
    - The CNN Legal Moves Variant incorporated multiple inputs for board data and legal move masks:
        - Board Input:
            - Conv2D layers to extract spatial relationships.
            - MaxPooling2D layers for dimensionality reduction.
            - Flattened output passed through dense layers.
        - Legal Moves Mask Input:
            - Multiplied with the model’s raw logits to enforce legal move predictions.
        - Combined Network:
            - The model’s output was restricted to legal moves using a softmax layer applied over the masked logits.
- **Training and Results**:
    - The model was trained for 30 epochs on 15k games from advanced and master players.
      
    | Dataset         | Loss   | Accuracy |
    |-----------------|--------|----------|
    | **Base Model**  | 7.9835 | 3.47%    |
  
- **Insights and Limitations**:
    - Improvements:
        - By focusing only on legal moves, the model avoided nonsensical predictions, ensuring all outputs adhered to chess rules.
    - Challenges:
        - The model’s accuracy remained very low (3.47%), indicating difficulty in learning optimal moves even when restricted to legal options.
        - The sequential and strategic nature of chess still posed significant challenges, as the CNN evaluated moves independently, without understanding the broader game context.
    - Key Takeaway:
        - The inability of the CNN to grasp the sequential complexities of chess led to the exploration of transformer-based models. Transformers, with their ability to analyze entire game sequences, were identified as a more promising architecture.

### Performance Summary

See the performance metrics of each model, highlighting their accuracy and loss across different datasets and training configurations.

Loss and accuracy are metrics used to evaluate the performance of machine learning models:

- **Loss**: Represents how far the model's predictions are from the actual outcomes. Lower loss values indicate better performance during training and testing.
- **Accuracy**: Measures the percentage of correct predictions made by the model. Higher accuracy reflects better alignment with the expected results.

| Model                          | Dataset ELO    | Games Trained | Loss   | Accuracy  |
|--------------------------------|----------------|---------------|--------|-----------|
| **DNN**                        | <1000          | 5k            | 5.9137 | 9.77%     |
|                                | 1000–1500      | 5k            | 5.7089 | 9.99%     |
|                                | 1500–2000      | 5k            | 5.7176 | 9.41%     |
|                                | >2000          | 5k            | 5.7024 | 8.83%     |
| **CNN**                        | <1000          | 5k            | 5.0660 | 10.77%    |
|                                | 1000–1500      | 5k            | 5.0810 | 10.46%    |
|                                | 1500–2000      | 5k            | 5.1264 | 10.24%    |
|                                | >2000          | 5k            | 5.1905 | 9.66%     |
| **CNN Unified**                | >1500          | 15k           | 5.2731 | 8.95%     |
| **CNN Legal Moves**            | >1500          | 15k           | 7.9835 | 3.47%     |
| **Base Transformer**           | All            | 15k           | 0.8449 | 84.69%    |
|                                | All            | 50k           | 0.5491 | 89.10%    |
| **Fine-Tuned Transformer**     | <1000          | 15k           | 0.4100 | 91.81%    |
|                                | 1000–1500      | 15k           | 0.4782 | 90.36%    |
|                                | 1500–2000      | 15k           | 0.5559 | 88.88%    |
|                                | >2000          | 15k           | 0.6249 | 87.61%    |

### Challenges & Learnings

Key observations and insights from the fine-tuned transformer models across different ELO ranges:

1. **Novice and Intermediate Bots**:
    - These bots effectively simulate their ELO range, demonstrating a solid understanding of legal moves and basic chess strategy.
    - Blunders and missed opportunities occur at an acceptable rate, making their playstyle feel realistic for their skill level.
2. **Advanced and Master Bots**:
    - While these bots perform well in identifying strong moves, they lack foresight for multi-turn strategies.
    - Their inability to adhere to specific chess openings can detract from their perceived ELO accuracy.
    - By integrating an alpha-beta pruning depth search during inference, the bots better evaluate moves with future consequences.
3. **ELO Consistency Issues**:
    - Advanced and master bots occasionally miss optimal moves, preventing them from fully reaching a 2000+ ELO standard.
    - The current approach trains models to predict the next best move based on game outcomes, but it fails to account for individual move quality.
    - Proposed Solution: Extend PGN labeling to include evaluations of individual moves, helping the model understand which moves contribute positively or negatively to game outcomes.
4. **Complexity of Chess Modeling**:
    - Chess requires a nuanced understanding of rules, strategy, and foresight. A model that only memorizes moves fails to capture the depth of the game.
    - The best-performing chess models likely require a hybrid approach of supervised and unsupervised learning to address all key aspects of chess:
        - Legal move generation (rules).
        - Positional understanding and strategy (supervised learning).
        - Long-term planning and foresight (unsupervised or reinforcement learning).

This iterative process of model development and evaluation provided invaluable insights into the complexities of chess modeling, ultimately guiding the implementation of a robust transformer-based solution capable of simulating realistic gameplay across all skill levels.

## Future Features

Here are some exciting features planned for future development:

1. **Expanded Training Dataset**: Increase the training size to over 100k games for each fine-tuned model, enabling more human-like gameplay and improved generalization across ELO ranges.
2. **Reinforcement Learning**: Further refine the master-level bot by incorporating reinforcement learning techniques, enabling it to learn advanced strategies and adapt dynamically to different playstyles.
3. **Enhanced Chess Move Evaluation**: Integrate a chess evaluator to assess and rate moves in PGN games. Include these evaluations in the transformer tensor to help the model distinguish between good and bad moves, particularly for high-level ELO play.
4. **Integrated Chess Phases**: Adapt the playstyle to react to the different stages of the game. Evaluate and split PGN games into the three main phases of chess: opening, middlegame, and endgame. Use this data to help the model determine the best strategy for each phase of the game.

## Contributing

If you want to contribute to this project, feel free to open an issue or submit a pull request. Any contributions, from bug fixes to new features, are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
