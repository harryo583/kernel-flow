#include <iostream>
#include <vector>
#include <fftw3.h>
#include "FFTWrapper.h"

enum class Padding {
    ZERO,       // pads the image or feature map with zeros
    CONSTANT,   // pads with a user-specified constant
    SAME,       // ensures that the output size is the same as the input size
    REPLICATE,  // pads by replicating the edge pixels of the image or feature map
    REFLECT,    // pads by reflecting border pixels
};

// 1D Padding function
static void pad1D(const std::vector<float>& input, std::vector<float>& output, Padding padding) {
    int len = input.size();         // length of input
    int klen = output.size() - len; // length of kernel

    if (padding == Padding::ZERO) {
        return;
    } else if (padding == Padding::REPLICATE) {
        for (int i = 0; i < len; ++i) {
            output[klen / 2 + i] = input[i];
        }
        output[0] = input[0];
        output[output.size() - 1] = input[len - 1];
        return;
    } else if (padding == Padding::REFLECT) {
        // Reflect padding
        for (int i = 0; i < len; ++i) {
            output[klen / 2 + i] = input[i];
        }
        for (int i = 0; i < klen / 2; ++i) {
            output[i] = input[klen / 2 - i];
            output[output.size() - 1 - i] = input[len - 1 - i];
        }
    } else {
        throw std::invalid_argument("Unsupported padding type");
    }
}

// 2D Padding function
static void pad2D(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, Padding padding) {
    int rows = input.size();               // height of input image
    int cols = input[0].size();            // width of input image
    int krows = output.size() - rows;      // height of kernel
    int kcols = output[0].size() - cols;   // width of kernel

    if (padding == Padding::ZERO) {
        return;
    } else if (padding == Padding::REPLICATE) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[krows / 2 + i][kcols / 2 + j] = input[i][j];
            }
        }
        // Replicate edges (top and bottom rows)
        for (int j = 0; j < output[0].size(); ++j) {
            output[0][j] = output[krows / 2][j];
            output[output.size() - 1][j] = output[output.size() - krows / 2 - 1][j];
        }
        // Replicate edges (left and right columns)
        for (int i = 0; i < output.size(); ++i) {
            output[i][0] = output[i][kcols / 2];
            output[i][output[0].size() - 1] = output[i][output[0].size() - kcols / 2 - 1];
        }
        return;
    } else if (padding == Padding::REFLECT) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[krows / 2 + i][kcols / 2 + j] = input[i][j];
            }
        }
        // Reflect edges (top and bottom rows)
        for (int j = 0; j < output[0].size(); ++j) {
            output[0][j] = output[1][j];
            output[output.size() - 1][j] = output[output.size() - 2][j];
        }
        // Reflect edges (left and right columns)
        for (int i = 0; i < output.size(); ++i) {
            output[i][0] = output[i][1];
            output[i][output[0].size() - 1] = output[i][output[0].size() - 2];
        }
        return;
    } else {
        throw std::invalid_argument("Unsupported padding type");
    }
}

// Main 2D convolution using FFT
std::vector<std::vector<float>> convolve(const std::vector<std::vector<float>>& image, const std::vector<std::vector<float>>& kernel, int stride, Padding padding) {
    int image_rows = image.size();
    int image_cols = image[0].size();
    int kernel_rows = kernel.size();
    int kernel_cols = kernel[0].size();
    
    int padded_rows = image_rows + kernel_rows - 1;
    int padded_cols = image_cols + kernel_cols - 1;
    
    std::vector<std::vector<float>> padded_image(padded_rows, std::vector<float>(padded_cols, 0));
    pad2D(image, padded_image, padding);
    std::vector<std::vector<float>> output_image(image_rows, std::vector<float>(image_cols, 0));
    
    FFTWrapper fftWrapper;

    convolve2D(padded_image, kernel, output_image, fftWrapper);

    return output_image;
}


static void convolve1D(const std::vector<float>& input_signal, 
                       const std::vector<float>& kernel, 
                       std::vector<float>& output_signal, 
                       FFTWrapper& fftWrapper) 
{
    int input_size = input_signal.size();
    int kernel_size = kernel.size();
    
    // Determine the padded size (usually the next power of two for efficient FFT)
    int padded_size = input_size + kernel_size - 1;

    // Prepare FFTW complex arrays for the input and kernel
    fftwf_complex *in_signal = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * padded_size);
    fftwf_complex *in_kernel = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * padded_size);
    fftwf_complex *out_signal = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * padded_size);

    std::vector<float> padded_input(padded_size, 0);
    std::vector<float> padded_kernel(padded_size, 0);
    
    for (int i = 0; i < input_size; ++i) {
        padded_input[i] = input_signal[i];
    }
    
    for (int i = 0; i < kernel_size; ++i) {
        padded_kernel[i] = kernel[i];
    }

    // Perform FFT on the padded input and kernel
    if (fftWrapper.performFFT1D(padded_input.data(), in_signal, padded_size, Flags::FFTW_ESTIMATE) != FFTStatus::SUCCESS) {
        throw std::runtime_error("FFT failed for input");
    }
    if (fftWrapper.performFFT1D(padded_kernel.data(), in_kernel, padded_size, Flags::FFTW_ESTIMATE) != FFTStatus::SUCCESS) {
        throw std::runtime_error("FFT failed for kernel");
    }

    // Perform element-wise multiplication in frequency domain
    for (int i = 0; i < padded_size; ++i) {
        out_signal[i][0] = in_signal[i][0] * in_kernel[i][0] - in_signal[i][1] * in_kernel[i][1];
        out_signal[i][1] = in_signal[i][0] * in_kernel[i][1] + in_signal[i][1] * in_kernel[i][0];
    }

    // Perform inverse FFT
    std::vector<float> convolved_signal(padded_size, 0);
    if (fftWrapper.performInverseFFT1D(out_signal, convolved_signal.data(), padded_size, Flags::FFTW_ESTIMATE) != FFTStatus::SUCCESS) {
        throw std::runtime_error("Inverse FFT failed");
    }

    // Crop the result to the expected output size
    output_signal.resize(input_size);
    for (int i = 0; i < input_size; ++i) {
        output_signal[i] = convolved_signal[i];
    }

    fftwf_free(in_signal);
    fftwf_free(in_kernel);
    fftwf_free(out_signal);
}

static void convolve2D(const std::vector<std::vector<float>>& padded_image,
                       const std::vector<std::vector<float>>& kernel,
                       std::vector<std::vector<float>>& output_image,
                       FFTWrapper& fftWrapper)
{
    int padded_rows = padded_image.size();
    int padded_cols = padded_image[0].size();
    int krows = kernel.size();
    int kcols = kernel[0].size();

    // Prepare FFTW complex arrays for the image and kernel
    fftwf_complex *in_image = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * padded_rows * padded_cols);
    fftwf_complex *in_kernel = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * krows * kcols);
    fftwf_complex *out_image = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * padded_rows * padded_cols);

    // Prepare input data for FFT
    for (int i = 0; i < padded_rows; ++i) {
        for (int j = 0; j < padded_cols; ++j) {
            in_image[i * padded_cols + j][0] = padded_image[i][j];
            in_image[i * padded_cols + j][1] = 0;
        }
    }

    for (int i = 0; i < krows; ++i) {
        for (int j = 0; j < kcols; ++j) {
            in_kernel[i * kcols + j][0] = kernel[i][j];
            in_kernel[i * kcols + j][1] = 0;
        }
    }

    // Perform FFT on the image and kernel
    if (fftWrapper.performFFT2D(reinterpret_cast<float*>(in_image), in_image, padded_rows, padded_cols, Flags::FFTW_ESTIMATE) != FFTStatus::SUCCESS) {
        throw std::runtime_error("FFT failed for image");
    }
    if (fftWrapper.performFFT2D(reinterpret_cast<float*>(in_kernel), in_kernel, krows, kcols, Flags::FFTW_ESTIMATE) != FFTStatus::SUCCESS) {
        throw std::runtime_error("FFT failed for kernel");
    }

    // Perform element-wise multiplication in frequency domain
    for (int i = 0; i < padded_rows * padded_cols; ++i) {
        out_image[i][0] = in_image[i][0] * in_kernel[i][0] - in_image[i][1] * in_kernel[i][1];
        out_image[i][1] = in_image[i][0] * in_kernel[i][1] + in_image[i][1] * in_kernel[i][0];
    }

    // Perform inverse FFT
    if (fftWrapper.performInverseFFT2D(out_image, reinterpret_cast<float*>(in_image), padded_rows, padded_cols, Flags::FFTW_ESTIMATE) != FFTStatus::SUCCESS) {
        throw std::runtime_error("Inverse FFT failed");
    }

    int output_rows = output_image.size();
    int output_cols = output_image[0].size();
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            output_image[i][j] = in_image[(i * output_cols + j)] / (padded_rows * padded_cols);
        }
    }

    fftwf_free(in_image);
    fftwf_free(in_kernel);
    fftwf_free(out_image);
}