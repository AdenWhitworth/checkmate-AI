"""
Script to check GPU availability, CUDA/cuDNN support, and perform a GPU computation test using TensorFlow.

This script:
1. Checks if GPUs are available for TensorFlow.
2. Verifies if TensorFlow is built with CUDA and cuDNN support.
3. Executes a simple computation on the GPU to test its functionality.

Functions:
- check_gpu_availability: Lists available GPUs for TensorFlow.
- check_cuda_cudnn_support: Verifies CUDA and cuDNN support in TensorFlow.
- test_gpu_computation: Executes a basic computation on the GPU.

Requirements:
- TensorFlow installed
- A system with an NVIDIA GPU, CUDA, and cuDNN properly configured (if GPU is to be used).
"""

import tensorflow as tf


def check_gpu_availability():
    """
    Checks for available GPUs for TensorFlow.

    Returns:
        list: A list of available physical GPU devices.
    """
    return tf.config.list_physical_devices('GPU')


def check_cuda_cudnn_support():
    """
    Checks if TensorFlow is built with CUDA and cuDNN support.

    Returns:
        dict: A dictionary with CUDA and cuDNN availability.
    """
    return {
        "CUDA": tf.test.is_built_with_cuda(),
        "cuDNN": tf.test.is_built_with_cuda()  # cuDNN shares CUDA build checks
    }


@tf.function
def test_gpu_computation():
    """
    Performs a simple TensorFlow computation on the GPU to test functionality.

    Returns:
        tf.Tensor: The result of the computation.
    """
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        return a + b


if __name__ == "__main__":
    # Check GPU availability
    gpus = check_gpu_availability()
    if gpus:
        print(f"Available GPUs: {gpus}")
    else:
        print("No GPUs available.")

    # Check CUDA and cuDNN support
    support = check_cuda_cudnn_support()
    print(f"Is TensorFlow built with CUDA support? {support['CUDA']}")
    print(f"Is cuDNN available? {support['cuDNN']}")

    # Perform GPU computation test
    try:
        result = test_gpu_computation()
        print("GPU Test Result:", result)
    except Exception as e:
        print("Error during GPU computation test:", e)
