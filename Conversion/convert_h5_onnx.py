"""
Module to convert a TensorFlow Keras model to ONNX format.

This script provides reusable functions to:
1. Load a TensorFlow model from a specified file.
2. Convert the model to ONNX format using tf2onnx.
3. Save the converted ONNX model to a specified file.

Requirements:
- TensorFlow (tf)
- tf2onnx

Usage:
- Call `convert_and_save_model()` with the TensorFlow model path and output ONNX path.
"""

import tf2onnx
import tensorflow as tf

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a TensorFlow Keras model from a file.

    Args:
        model_path (str): Path to the TensorFlow model file (.h5).

    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model. Ensure the path is correct: {model_path}\n{e}")

def convert_to_onnx(model: tf.keras.Model, opset_version: int = 13) -> bytes:
    """
    Convert a TensorFlow Keras model to ONNX format.

    Args:
        model (tf.keras.Model): The TensorFlow Keras model to convert.
        opset_version (int): The ONNX opset version to use for compatibility.

    Returns:
        bytes: The serialized ONNX model as a byte string.
    """
    try:
        # Define the input signature for the model
        input_signature = (tf.TensorSpec((None, model.input_shape[1]), tf.float32),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=opset_version)
        print(f"Model converted to ONNX format using opset version {opset_version}")
        return model_proto.SerializeToString()
    except Exception as e:
        raise RuntimeError(f"Failed to convert model to ONNX: {e}")

def save_onnx_model(onnx_model: bytes, output_path: str) -> None:
    """
    Save the ONNX model to a file.

    Args:
        onnx_model (bytes): The serialized ONNX model as a byte string.
        output_path (str): Path to save the ONNX model.
    """
    try:
        with open(output_path, "wb") as f:
            f.write(onnx_model)
        print(f"ONNX model saved to {output_path}")
    except Exception as e:
        raise IOError(f"Failed to save ONNX model to {output_path}\n{e}")

def convert_and_save_model(model_path: str, output_path: str, opset_version: int = 13) -> None:
    """
    Complete workflow to load a TensorFlow model, convert it to ONNX, and save the ONNX model.

    Args:
        model_path (str): Path to the TensorFlow model file (.h5).
        output_path (str): Path to save the ONNX model.
        opset_version (int): The ONNX opset version to use for conversion.
    """
    print("Starting TensorFlow to ONNX conversion process...")
    model = load_model(model_path)
    onnx_model = convert_to_onnx(model, opset_version)
    save_onnx_model(onnx_model, output_path)
    print("Conversion process completed successfully.")

# Example usage:
if __name__ == "__main__":
    # Paths to the TensorFlow model and output ONNX file
    model_path = "fine_tuned_elo_2000_plus.h5"
    output_path = "model_elo_2000_plus.onnx"

    # Convert and save the model
    convert_and_save_model(model_path, output_path)