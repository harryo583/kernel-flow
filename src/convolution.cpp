#include <iostream>
#include <vector>
#include <fftw3.h>
#include "FFTWrapper.h"
#include <iomanip>

enum class Padding {
    VALID,      // no padding
    ZERO,       // pads the image or feature map with zeros
    CONSTANT,   // pads with a user-specified constant
    REPLICATE,  // pads by replicating the edge pixels of the image or feature map
    REFLECT,    // pads by reflecting border pixels
};

// 1D Padding function
static void pad1D(const std::vector<float>& input,
                  std::vector<float>& output,
                  Padding padding,
                  float pad_val = 0.0) {
    int len = input.size();
    int klen = output.size() - len;
    int pad_left = klen / 2;
    int pad_right = klen - pad_left;

    // Copy input data
    for (int i = 0; i < len; i++) output[pad_left + i] = input[i];

    // Apply padding
    if (padding == Padding::ZERO) {
        std::fill(output.begin(), output.begin() + pad_left, 0);
        std::fill(output.end() - pad_right, output.end(), 0);
    } else if (padding == Padding::CONSTANT) {
        std::fill(output.begin(), output.begin() + pad_left, pad_val);
        std::fill(output.end() - pad_right, output.end(), pad_val);
    } else if (padding == Padding::REPLICATE) {
        std::fill(output.begin(), output.begin() + pad_left, input[0]);
        std::fill(output.end() - pad_right, output.end(), input[len - 1]);
    } else if (padding == Padding::REFLECT) {
        for (int i = 0; i < pad_left; i++) output[i] = input[pad_left - i - 1];
        for (int i = 0; i < pad_right; i++) output[output.size() - 1 - i] = input[len - pad_right + i];
    } else {
        throw std::invalid_argument("Unsupported padding type");
    }
}

// 2D Padding function
static void pad2D(const std::vector<std::vector<float>>& input, 
                  std::vector<std::vector<float>>& output, 
                  Padding padding, float pad_val = 0.0) {
    int rows = input.size();
    int cols = input[0].size();
    int krows = output.size() - rows;
    int kcols = output[0].size() - cols;
    int pad_top = krows / 2;
    int pad_left = kcols / 2;

    // Copy input data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[pad_top + i][pad_left + j] = input[i][j];
        }
    }

    // Apply padding
    if (padding == Padding::ZERO) {
        for (std::vector<std::vector<float>>::size_type i = 0; i < output.size(); i++) {
            for (std::vector<float>::size_type j = 0; j < output[i].size(); j++) {
                if (i < pad_top || i >= rows + pad_top || j < pad_left || j >= cols + pad_left) {
                    output[i][j] = 0;
                }
            }
        }
    } else if (padding == Padding::CONSTANT) {
        for (std::vector<std::vector<float>>::size_type i = 0; i < output.size(); i++) {
            for (std::vector<float>::size_type j = 0; j < output[i].size(); j++) {
                if (i < pad_top || i >= rows + pad_top || j < pad_left || j >= cols + pad_left) {
                    output[i][j] = pad_val;
                }
            }
        }
    } else if (padding == Padding::REPLICATE) {
        for (int i = 0; i < pad_top; i++) {
            for (std::vector<float>::size_type j = 0; j < output[0].size(); j++) {
                output[i][j] = input[0][j];
                output[output.size() - 1 - i][j] = input[rows - 1][j];
            }
        }
        for (int j = 0; j < pad_left; j++) {
            for (std::vector<std::vector<float>>::size_type i = 0; i < output.size(); i++) {
                output[i][j] = input[i][0];
                output[i][output[0].size() - 1 - j] = input[i][cols - 1];
            }
        }
    } else if (padding == Padding::REFLECT) {
        for (int i = 0; i < pad_top; i++) {
            for (std::vector<float>::size_type j = 0; j < output[0].size(); j++) {
                output[i][j] = input[pad_top - i - 1][j];
                output[output.size() - 1 - i][j] = input[rows - pad_top + i][j];
            }
        }
        for (int j = 0; j < pad_left; j++) {
            for (std::vector<std::vector<float>>::size_type i = 0; i < output.size(); i++) {
                output[i][j] = input[i][pad_left - j - 1];
                output[i][output[0].size() - 1 - j] = input[i][cols - pad_left + j];
            }
        }
    } else {
        throw std::invalid_argument("Unsupported padding type");
    }
}


static void convolve1D(const std::vector<float>& input_signal, 
                       const std::vector<float>& kernel,
                       std::vector<float>& output_signal,
                       FFTWrapper& fftWrapper) {
    int signal_size = input_signal.size();
    int kernel_size = kernel.size();

    int padded_size = signal_size + kernel_size - 1;

    std::vector<float> padded_signal(padded_size, 0);
    std::vector<float> padded_kernel(padded_size, 0);

    // Copy signal and kernel
    std::copy(input_signal.begin(), input_signal.end(), padded_signal.begin());
    std::copy(kernel.begin(), kernel.end(), padded_kernel.begin());

    std::vector<std::complex<float>> fft_signal(padded_size);
    std::vector<std::complex<float>> fft_kernel(padded_size);
    std::vector<std::complex<float>> fft_result(padded_size);

    fftWrapper.performFFT1D(padded_signal, fft_signal, Flags::ESTIMATE);
    fftWrapper.performFFT1D(padded_kernel, fft_kernel, Flags::ESTIMATE);

    for (int i = 0; i < padded_size; ++i) {
        fft_result[i] = fft_signal[i] * fft_kernel[i];
    }

    fftWrapper.performInverseFFT1D(fft_result, output_signal, Flags::ESTIMATE);
}

static void convolve2D(const std::vector<std::vector<float>>& input_image,
                       const std::vector<std::vector<float>>& kernel,
                       std::vector<std::vector<float>>& output_image,
                       FFTWrapper& fftWrapper) {
    int image_rows = input_image.size();
    int image_cols = input_image[0].size();
    int kernel_rows = kernel.size();
    int kernel_cols = kernel[0].size();

    int padded_rows = image_rows + kernel_rows - 1;
    int padded_cols = image_cols + kernel_cols - 1;

    std::vector<std::vector<float>> padded_image(padded_rows, std::vector<float>(padded_cols, 0));
    std::vector<std::vector<float>> padded_kernel(padded_rows, std::vector<float>(padded_cols, 0));

    // Copy image and kernel
    for (int i = 0; i < image_rows; ++i) {
        for (int j = 0; j < image_cols; ++j) {
            padded_image[i + (kernel_rows - 1) / 2][j + (kernel_cols - 1) / 2] = input_image[i][j];
        }
    }

    for (int i = 0; i < kernel_rows; ++i) {
        for (int j = 0; j < kernel_cols; ++j) {
            padded_kernel[i][j] = kernel[kernel_rows - i - 1][kernel_cols - j - 1];
        }
    }

    std::vector<std::vector<std::complex<float>>> fft_image(padded_rows, std::vector<std::complex<float>>(padded_cols));
    std::vector<std::vector<std::complex<float>>> fft_kernel(padded_rows, std::vector<std::complex<float>>(padded_cols));
    std::vector<std::vector<std::complex<float>>> fft_result(padded_rows, std::vector<std::complex<float>>(padded_cols));

    for (int i = 0; i < padded_rows; ++i) {
        fftWrapper.performFFT1D(padded_image[i], fft_image[i], Flags::ESTIMATE);
        fftWrapper.performFFT1D(padded_kernel[i], fft_kernel[i], Flags::ESTIMATE);
    }

    for (int i = 0; i < padded_rows; ++i) {
        for (int j = 0; j < padded_cols; ++j) {
            fft_result[i][j] = fft_image[i][j] * fft_kernel[i][j];
        }
    }

    for (int i = 0; i < padded_rows; ++i) {
        fftWrapper.performInverseFFT1D(fft_result[i], output_image[i], Flags::ESTIMATE);
    }
}


// Overloaded function for 1D convolution
std::vector<float> convolve(const std::vector<float>& signal,
                            const std::vector<float>& kernel,
                            int stride,
                            Padding padding,
                            float pad_val = 0.0) {
    
    FFTWrapper fftWrapper;

    int input_size = signal.size();
    int kernel_size = kernel.size();

    if (padding != Padding::VALID) {
        int padded_size = input_size + kernel_size - 1;
        std::vector<float> padded_input(padded_size, 0);
        std::vector<float> output_signal(input_size);
        pad1D(signal, padded_input, padding, pad_val);
        convolve1D(padded_input, kernel, output_signal, fftWrapper);
        return output_signal;
    } else {
        auto padded_input = signal;
        std::vector<float> output_signal(input_size - kernel_size + 1);
        convolve1D(padded_input, kernel, output_signal, fftWrapper);
        return output_signal;
    }
}

// Overloaded function for 2D convolution
std::vector<std::vector<float>> convolve(const std::vector<std::vector<float>>& image,
                                         const std::vector<std::vector<float>>& kernel,
                                         int stride,
                                         Padding padding,
                                         float pad_val = 0.0) {

    int image_rows = image.size();
    int image_cols = image[0].size();
    int kernel_rows = kernel.size();
    int kernel_cols = kernel[0].size();
    
    if (padding != Padding::VALID) {
        int padded_rows = image_rows + kernel_rows - 1;
        int padded_cols = image_cols + kernel_cols - 1;
        
        std::vector<std::vector<float>> padded_image(padded_rows, std::vector<float>(padded_cols, 0));
        std::vector<std::vector<float>> output_image(image_rows, std::vector<float>(image_cols, 0));
        pad2D(image, padded_image, padding, pad_val);
        FFTWrapper fftWrapper;
        convolve2D(padded_image, kernel, output_image, fftWrapper);
        return output_image;
    } else {
        auto padded_image = image;
        std::vector<std::vector<float>> output_image(image_rows - kernel_rows + 1, std::vector<float>(image_cols - kernel_cols + 1, 0));
        FFTWrapper fftWrapper;
        convolve2D(padded_image, kernel, output_image, fftWrapper);
        return output_image;
    }
}




// Function to print a 1D vector
void print1DVector(const std::vector<float>& vec) {
    for (const auto& val : vec) {
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    }
    std::cout << std::endl;
}

// Function to print a 2D vector
void print2DVector(const std::vector<std::vector<float>>& mat) {
    for (const auto& row : mat) {
        for (const auto& val : row) {
            std::cout << std::fixed << std::setprecision(2) << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    FFTWrapper fftWrapper;

    // Test 1D Convolution
    std::vector<float> signal = {1, 2, 3, 4};
    std::vector<float> kernel = {1, 0, -1};
    
    std::cout << "1D Convolution Test:" << std::endl;

    std::vector<float> output_signal = convolve(signal, kernel, 1, Padding::ZERO);
    std::cout << "Output with ZERO padding:" << std::endl;
    print1DVector(output_signal);

    output_signal = convolve(signal, kernel, 1, Padding::CONSTANT, 2.0);
    std::cout << "Output with CONSTANT padding (value 2.0):" << std::endl;
    print1DVector(output_signal);

    output_signal = convolve(signal, kernel, 1, Padding::REPLICATE);
    std::cout << "Output with REPLICATE padding:" << std::endl;
    print1DVector(output_signal);

    output_signal = convolve(signal, kernel, 1, Padding::REFLECT);
    std::cout << "Output with REFLECT padding:" << std::endl;
    print1DVector(output_signal);

    // Test 2D Convolution
    std::vector<std::vector<float>> image = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::vector<float>> kernel2D = {
        {1, 0},
        {-1, 1}
    };

    std::cout << "\n2D Convolution Test:" << std::endl;

    std::vector<std::vector<float>> output_image = convolve(image, kernel2D, 1, Padding::ZERO);
    std::cout << "Output with ZERO padding:" << std::endl;
    print2DVector(output_image);

    output_image = convolve(image, kernel2D, 1, Padding::CONSTANT, 2.0);
    std::cout << "Output with CONSTANT padding (value 2.0):" << std::endl;
    print2DVector(output_image);

    output_image = convolve(image, kernel2D, 1, Padding::REPLICATE);
    std::cout << "Output with REPLICATE padding:" << std::endl;
    print2DVector(output_image);

    output_image = convolve(image, kernel2D, 1, Padding::REFLECT);
    std::cout << "Output with REFLECT padding:" << std::endl;
    print2DVector(output_image);

    return 0;
}