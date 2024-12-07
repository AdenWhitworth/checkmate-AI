import tf2onnx
import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model("fine_tuned_elo_2000_plus.h5")

# Convert the model to ONNX format
spec = (tf.TensorSpec((None, model.input_shape[1]), tf.float32),)  # Adjust input shape if needed
output_path = "model_elo_2000_plus.onnx"

# Convert and save
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"ONNX model saved to {output_path}")
