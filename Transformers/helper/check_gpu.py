import tensorflow as tf

# Check GPU availability
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Test CUDA availability
print("Is TensorFlow built with CUDA support?", tf.test.is_built_with_cuda())

# Test cuDNN availability
print("Is cuDNN available?", tf.test.is_built_with_cuda())

# GPU computation test
@tf.function
def test_gpu():
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        return a + b

try:
    print("GPU Test Result:", test_gpu())
except Exception as e:
    print("Error:", e)

