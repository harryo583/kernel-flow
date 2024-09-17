# kernel-flow

**kernel-flow** is a high-performance image processing library written in C++, designed for efficient use in both C++ and Python environments. The library excels in large-scale convolution operations, making it ideal for tasks such as those used in Convolutional Neural Networks (CNNs).

**Status: Under Construction**

This library is currently under development. Features are being implemented and tested. Please check back later for updates.

## Key Features

### 1. Fast Convolution Using Fast Fourier Transform (FFT)

kernel-flow leverages the power of the Fast Fourier Transform (FFT) to perform convolution operations more efficiently than standard spatial-domain methods. This is particularly advantageous for large convolution operations where traditional methods can be too slow.

- **Efficient FFT-based convolution**: Reduces computation time for large convolutions.
- **High scalability**: Suitable for large datasets or high-dimensional inputs.

### 2. Parallelized Execution with OpenMP

To fully utilize modern multi-core processors, kernel-flow parallelizes FFT computation using OpenMP. This enables multi-threading, allowing multiple CPU cores to work concurrently and significantly speeding up the convolution process.

- **Multi-threaded execution**: Optimized for multi-core CPUs.
- **Reduced computation time**: Takes full advantage of system hardware for faster performance.

### 3. Optional Integration with CUDA

For users looking for even greater performance, kernel-flow will provide optional CUDA integration. This allows FFT computations to be offloaded to GPUs, making the library suitable for extremely large-scale convolution tasks.

- **GPU acceleration (optional)**: Harness the power of CUDA-enabled GPUs for even faster computation.
- **Efficient for large-scale operations**: Dramatically reduces computation time for heavy workloads.

## License

This project is licensed under the MIT License.
