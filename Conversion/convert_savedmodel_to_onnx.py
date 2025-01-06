"""
Script to convert TensorFlow SavedModel directories to ONNX format.

This script iterates over all subdirectories in a given input directory, checks if each subdirectory
is a valid SavedModel (contains `saved_model.pb`), and converts it to ONNX format using tf2onnx.

Requirements:
- TensorFlow
- tf2onnx installed in the Python environment.

Functions:
- convert_savedmodel_to_onnx: Main function to perform the conversion for all models in a directory.

Usage:
- Run this script directly to convert SavedModels in `node_models` to ONNX format in `onnx_models`.
"""

import os
import sys
import subprocess

def convert_savedmodel_to_onnx(input_dir: str, output_dir: str) -> None:
    """
    Convert TensorFlow SavedModel directories to ONNX format.

    This function scans the input directory for subdirectories containing SavedModels
    (`saved_model.pb`), converts each to ONNX format using tf2onnx, and saves the
    resulting ONNX models to the output directory.

    Args:
        input_dir (str): Path to the directory containing SavedModel subdirectories.
        output_dir (str): Path to save the converted ONNX models.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        Exception: For any unexpected errors during the conversion process.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured: {output_dir}")

    # Iterate over all subdirectories in the input directory
    for model_name in os.listdir(input_dir):
        model_path = os.path.join(input_dir, model_name)

        # Check if the directory contains a SavedModel (saved_model.pb)
        if os.path.isdir(model_path) and "saved_model.pb" in os.listdir(model_path):
            onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
            print(f"Converting {model_path} to {onnx_path}...")

            # Use tf2onnx to convert the model
            try:
                subprocess.run(
                    [
                        sys.executable, "-m", "tf2onnx.convert",
                        "--saved-model", model_path,
                        "--output", onnx_path
                    ],
                    check=True,
                    env=os.environ  # Use current environment
                )
                print(f"Successfully converted {model_name} to {onnx_path}.")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {model_name}: {e}")
            except Exception as e:
                print(f"Unexpected error while converting {model_name}: {e}")
        else:
            print(f"Skipping {model_name}: Not a valid SavedModel directory.")

if __name__ == "__main__":
    # Define paths
    input_directory = "node_models"  # Path to the directory containing SavedModels
    output_directory = "onnx_models"  # Path to save the ONNX models

    # Convert models
    try:
        convert_savedmodel_to_onnx(input_directory, output_directory)
    except Exception as e:
        print(f"Failed to complete the conversion process: {e}")


