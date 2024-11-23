import os
from tensorflow.keras.models import load_model

# Paths for models and output directories
model_paths = {
    "less_1000_model": "models/less_1000_model.keras",
    "1000_1500_model": "models/1000_1500_model.keras",
    "1500_2000_model": "models/1500_2000_model.keras",
    "greater_2000_model": "models/greater_2000_model.keras"
}
output_dir = "node_models"  # Directory to save TensorFlow.js models

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over the models and convert them
for model_name, model_path in model_paths.items():
    print(f"Processing {model_name}...")

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        continue

    try:
        # Load the Keras model
        model = load_model(model_path)

        # Define the output path for the SavedModel format
        saved_model_path = os.path.join(output_dir, model_name)

        # Save the model in SavedModel format
        model.save(saved_model_path, save_format="tf")
        print(f"Model saved to {saved_model_path}")
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
