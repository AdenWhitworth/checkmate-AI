"""
Script to convert TensorFlow Keras models to SavedModel format for TensorFlow.js.

This script provides reusable functions to:
1. Verify the existence of model files.
2. Convert Keras models to TensorFlow SavedModel format.
3. Save the converted models to a specified output directory.

Requirements:
- TensorFlow
- Directory structure containing models in .keras format.

Usage:
- Call `convert_models_to_saved_model()` with the model paths and output directory.
"""

import os
from tensorflow.keras.models import load_model

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that the specified directory exists. Create it if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Ensured directory exists: {directory}")

def load_keras_model(model_path: str):
    """
    Load a TensorFlow Keras model from the specified file.

    Args:
        model_path (str): Path to the Keras model file.

    Returns:
        tf.keras.Model: The loaded Keras model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from {model_path}...")
    return load_model(model_path)

def save_model_to_saved_model(model, output_path: str) -> None:
    """
    Save a TensorFlow Keras model to SavedModel format.

    Args:
        model (tf.keras.Model): The Keras model to save.
        output_path (str): Directory path to save the model in SavedModel format.
    """
    print(f"Saving model to {output_path}...")
    model.save(output_path, save_format="tf")
    print(f"Model successfully saved to {output_path}")

def process_model_conversion(model_name: str, model_path: str, output_dir: str) -> None:
    """
    Load a Keras model and save it in SavedModel format.

    Args:
        model_name (str): Name of the model for display/logging purposes.
        model_path (str): Path to the Keras model file.
        output_dir (str): Directory to save the converted model.
    """
    try:
        model = load_keras_model(model_path)
        saved_model_path = os.path.join(output_dir, model_name)
        save_model_to_saved_model(model, saved_model_path)
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

def convert_models_to_saved_model(model_paths: dict, output_dir: str) -> None:
    """
    Convert multiple TensorFlow Keras models to SavedModel format.

    Args:
        model_paths (dict): Dictionary with model names as keys and paths as values.
        output_dir (str): Directory to save the converted models.
    """
    print("Starting model conversion process...")
    ensure_directory_exists(output_dir)

    for model_name, model_path in model_paths.items():
        print(f"Processing model: {model_name}")
        process_model_conversion(model_name, model_path, output_dir)

    print("Model conversion process completed successfully.")

# Example usage
if __name__ == "__main__":
    # Paths to Keras models and output directory
    model_paths = {
        "less_1000_model": "models/less_1000_model.keras",
        "1000_1500_model": "models/1000_1500_model.keras",
        "1500_2000_model": "models/1500_2000_model.keras",
        "greater_2000_model": "models/greater_2000_model.keras",
    }
    output_dir = "node_models"  # Directory to save TensorFlow.js models

    # Convert and save all models
    convert_models_to_saved_model(model_paths, output_dir)
