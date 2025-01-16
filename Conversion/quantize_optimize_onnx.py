"""
Script to optimize and dynamically quantize an ONNX model.

This script loads an ONNX model, validates it, applies optimizations using `onnxoptimizer`,
and performs dynamic quantization using `onnxruntime.quantization`. The optimized and
quantized models are saved to the specified output directory.

Requirements:
- ONNX
- ONNX Runtime
- ONNX Optimizer

Functions:
- optimize_and_quantize: Main function to optimize and quantize an ONNX model.

Usage:
- Replace `input_model_path` and `output_directory` with your model path and desired output directory.
"""

import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxoptimizer import optimize

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that the specified directory exists. Create it if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Ensured directory exists: {directory}")

def validate_onnx_model(model_path: str) -> onnx.ModelProto:
    """
    Load and validate an ONNX model.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        onnx.ModelProto: The validated ONNX model.

    Raises:
        FileNotFoundError: If the ONNX model file does not exist.
        onnx.checker.ValidationError: If the ONNX model fails validation.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    
    print(f"Loading ONNX model from: {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(f"Model validation passed for: {model_path}")
    return model

def save_optimized_model(model: onnx.ModelProto, output_path: str) -> None:
    """
    Optimize and save an ONNX model using ONNX Optimizer.

    Args:
        model (onnx.ModelProto): The ONNX model to optimize.
        output_path (str): Path to save the optimized model.
    """
    print(f"Optimizing model and saving to: {output_path}")
    optimized_model = optimize(model)
    onnx.save(optimized_model, output_path)
    print(f"Optimized model saved at: {output_path}")

def save_quantized_model(input_model_path: str, output_model_path: str) -> None:
    """
    Perform dynamic quantization on an ONNX model and save the quantized version.

    Args:
        input_model_path (str): Path to the input ONNX model.
        output_model_path (str): Path to save the quantized model.
    """
    print(f"Quantizing model and saving to: {output_model_path}")
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved at: {output_model_path}")

def optimize_and_quantize(model_path: str, output_dir: str) -> None:
    """
    Optimizes and dynamically quantizes an ONNX model.

    Args:
        model_path (str): Path to the input ONNX model.
        output_dir (str): Directory to save the optimized and quantized models.
    """
    ensure_directory_exists(output_dir)

    model = validate_onnx_model(model_path)

    optimized_model_path = os.path.join(output_dir, "optimized_model.onnx")
    save_optimized_model(model, optimized_model_path)

    quantized_model_path = os.path.join(output_dir, "optimized_model_quantized.onnx")
    save_quantized_model(optimized_model_path, quantized_model_path)

    print("Model optimization and quantization completed successfully.")

if __name__ == "__main__":
    input_model_path = "../Transformers/v5/models/checkpoints/onnx_model/model_final_with_outcome.onnx"
    output_directory = "../Transformers/v5/models/checkpoints/onnx_model"

    try:
        optimize_and_quantize(input_model_path, output_directory)
    except Exception as e:
        print(f"Error during optimization and quantization: {e}")