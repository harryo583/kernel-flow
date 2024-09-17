# kernel-flow
kernel-flow is a C++-based high-performance image processing library designed for use with C++ and Python. It is suited for large-scale convolution operations such as those used in Convolutional Neural Networks.

Key Features
1. Fast Convolution Using Fast Fourier Transform (FFT)
The library leverages the power of the Fast Fourier Transform (FFT) to perform convolution operations more efficiently than standard spatial-domain methods. This approach is especially beneficial for large convolution operations where traditional methods are too slow.
2. Parallelized Execution with OpenMP
To fully utilize modern multi-core processors, kernel-flow parallelizes the FFT computation using OpenMP. This allows multiple threads to work concurrently, reducing the time required to compute convolutions.
3. Optional Integration with CUDA
For users seeking even greater performance, kernel-flow offers optional integration with CUDA. By leveraging the power of GPUs, users can offload FFT computations to the GPU, drastically reducing the time needed for large convolution operations.
