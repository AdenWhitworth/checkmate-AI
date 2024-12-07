import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def optimize_and_quantize(model_path, output_dir):
    """
    Optimizes and dynamically quantizes an ONNX model.
    
    Args:
        model_path (str): Path to the input ONNX model.
        output_dir (str): Directory to save the optimized and quantized models.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and validate the ONNX model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(f"Model validation passed for: {model_path}")

    # Save the optimized model
    optimized_model_path = os.path.join(output_dir, "model_elo_2000_plus_optimized.onnx")
    from onnxoptimizer import optimize
    optimized_model = optimize(model)
    onnx.save(optimized_model, optimized_model_path)
    print(f"Optimized model saved at: {optimized_model_path}")

    # Dynamically quantize the optimized model
    quantized_model_path = os.path.join(output_dir, "model_elo_2000_plus_optimized_quantized.onnx")
    quantize_dynamic(
        model_input=optimized_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8,  # Quantize weights to INT8
    )
    print(f"Quantized model saved at: {quantized_model_path}")



input_model_path = "model_elo_2000_plus.onnx"  # Replace with your model file path
output_directory = "optimized_output_models"  # Directory to save outputs

optimize_and_quantize(input_model_path, output_directory)





