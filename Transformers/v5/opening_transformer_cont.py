#Epoch 10 to 37: 16s 5ms/step - loss: 1.6794 - accuracy: 0.4620 - val_loss: 1.6389 - val_accuracy: 0.4732
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
from sklearn.model_selection import train_test_split

# File paths
model_file = "models/checkpoints/model_final.h5"  # Previously trained model
updated_model_file = "models/checkpoints/model_final_updated.h5"  # Save updated model
fens_file = "models/checkpoints/fens.npy"
moves_file = "models/checkpoints/moves.npy"
labels_file = "models/checkpoints/labels.npy"
move_to_idx_file = "models/checkpoints/move_to_idx.json"

# Load preprocessed data
fens = np.load(fens_file)
moves = np.load(moves_file)
labels = np.load(labels_file)
with open(move_to_idx_file, "r") as f:
    move_to_idx = json.load(f)

# Load previously trained model
model = load_model(model_file)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    initial_lr = 1e-4
    decay_epochs = 10
    alpha = 0.1
    return initial_lr * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(np.pi * epoch / decay_epochs)))

# Callbacks
checkpoint = ModelCheckpoint("models/checkpoints/model_checkpoint_updated.h5", save_best_only=True, monitor="val_loss", mode="min")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
lr_schedule = LearningRateScheduler(lr_scheduler)

# Split the data
X_fens_train, X_fens_test, X_moves_train, X_moves_test, y_train, y_test = train_test_split(
    fens, moves, labels, test_size=0.2, random_state=42
)

# Continue training
print("Continuing model training...")
history = model.fit(
    [X_fens_train, X_moves_train], y_train,
    epochs=50,  # Extend training to a total of 50 epochs
    initial_epoch=10,  # Resume from the last completed epoch
    batch_size=32,
    validation_data=([X_fens_test, X_moves_test], y_test),
    callbacks=[checkpoint, early_stopping, lr_schedule]
)

# Save the updated model
model.save(updated_model_file)
print("Extended training complete and saved.")
