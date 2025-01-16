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
import os


def load_model(model_path: str, custom_objects=None) -> tf.keras.Model:
    """
    Load a TensorFlow Keras model from a file.

    Args:
        model_path (str): Path to the TensorFlow model file (.h5).
        custom_objects (dict): Optional dictionary of custom objects required for the model.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """
    try:
        if custom_objects:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model from {model_path}. Ensure the path is correct.\nError: {e}")


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
        # Define input signature
        input_signature = [tf.TensorSpec(shape=input_tensor.shape, dtype=input_tensor.dtype) for input_tensor in model.inputs]

        # Convert to ONNX
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            opset=opset_version,
            input_signature=input_signature
        )
        print(f"Model converted to ONNX format using opset version {opset_version}")
        return model_proto.SerializeToString()
    except Exception as e:
        raise RuntimeError(f"Failed to convert model to ONNX.\nError: {e}")


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
        raise IOError(f"Failed to save ONNX model to {output_path}\nError: {e}")


def convert_and_save_model(model_path: str, output_path: str, custom_objects=None, opset_version: int = 13) -> None:
    """
    Complete workflow to load a TensorFlow model, convert it to ONNX, and save the ONNX model.

    Args:
        model_path (str): Path to the TensorFlow model file (.h5).
        custom_objects (dict): Optional dictionary of custom objects required for the model.
        output_path (str): Path to save the ONNX model.
        opset_version (int): The ONNX opset version to use for conversion.
    """
    print("Starting TensorFlow to ONNX conversion process...")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = load_model(model_path, custom_objects)
    onnx_model = convert_to_onnx(model, opset_version)
    save_onnx_model(onnx_model, output_path)
    print("Conversion process completed successfully.")


if __name__ == "__main__":
    model_path = "../Transformers/v6/models/checkpoints3/model_midgame_final.h5"
    output_path = "../Transformers/v6/models/checkpoints3/onnx_model/model_midgame_final.onnx"

    def top_k_accuracy(y_true, y_pred, k=5):
        """
        Calculate the top-k categorical accuracy for sparse labels.

        Args:
            y_true: Ground truth labels (integer indices).
            y_pred: Predicted probabilities for each class.
            k (int): Number of top predictions to consider.

        Returns:
            tf.Tensor: Top-k accuracy as a scalar tensor.
        """
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

    convert_and_save_model(model_path, output_path, {"top_k_accuracy": top_k_accuracy})
