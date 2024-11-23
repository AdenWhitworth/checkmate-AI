import os
import sys
import subprocess

def convert_savedmodel_to_onnx(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

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
        else:
            print(f"Skipping {model_name}: Not a valid SavedModel directory.")

if __name__ == "__main__":
    # Define paths
    input_directory = "node_models"  # Path to the directory containing SavedModels
    output_directory = "onnx_models"  # Path to save the ONNX models

    # Convert models
    convert_savedmodel_to_onnx(input_directory, output_directory)

