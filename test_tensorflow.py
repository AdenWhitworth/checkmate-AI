#pip install -r requirements.txt
import tensorflow as tf
import numpy as np

# Define a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generate dummy data
x = np.random.random((100, 4))
y = np.random.randint(0, 3, (100,))

# Train the model
model.fit(x, y, epochs=5)

print("TensorFlow is working!")
