// /src/convolution.cpp

#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include "FFTWrapper.h"

// Padding enumeration (for boundary handling)
enum class Padding {
    VALID,      // no padding
    ZERO,       // pads with zeros
    CONSTANT,   // pads with a constant value
    REPLICATE,  // pads by replicating the edge values
    REFLECT,    // pads by reflecting border values
};

// -------------------- 1D Padding --------------------
// Given an input vector, write it into output so that the “input”
 // occupies the center and the borders are filled according to the chosen type.
static void pad1D(const std::vector<float>& input,
                  std::vector<float>& output,
                  Padding padding,
                  float pad_val = 0.0f) {
    int len = input.size();
    int totalPad = output.size() - len;
    int pad_left = totalPad / 2;
    int pad_right = totalPad - pad_left;
    
    // Copy the input into the center.
    std::copy(input.begin(), input.end(), output.begin() + pad_left);
    
    if (padding == Padding::ZERO) {
        std::fill(output.begin(), output.begin() + pad_left, 0.0f);
        std::fill(output.end() - pad_right, output.end(), 0.0f);
    }
    else if (padding == Padding::CONSTANT) {
        std::fill(output.begin(), output.begin() + pad_left, pad_val);
        std::fill(output.end() - pad_right, output.end(), pad_val);
    }
    else if (padding == Padding::REPLICATE) {
        std::fill(output.begin(), output.begin() + pad_left, input.front());
        std::fill(output.end() - pad_right, output.end(), input.back());
    }
    else if (padding == Padding::REFLECT) {
        for (int i = 0; i < pad_left; i++)
            output[i] = input[pad_left - i - 1];
        for (int i = 0; i < pad_right; i++)
            output[output.size() - 1 - i] = input[len - pad_right + i];
    }
    else {
        throw std::invalid_argument("Unsupported padding type");
    }
}

// -------------------- 2D Padding --------------------
static void pad2D(const std::vector<std::vector<float>>& input, 
                  std::vector<std::vector<float>>& output, 
                  Padding padding, float pad_val = 0.0f) {
    int rows = input.size();
    int cols = input[0].size();
    int totalPadRows = output.size() - rows;
    int totalPadCols = output[0].size() - cols;
    int pad_top = totalPadRows / 2;
    int pad_left = totalPadCols / 2;
    
    // Copy the input into the center.
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            output[pad_top + i][pad_left + j] = input[i][j];
    
    if (padding == Padding::ZERO) {
        // Top and bottom rows
        for (int i = 0; i < pad_top; i++)
            std::fill(output[i].begin(), output[i].end(), 0.0f);
        for (int i = output.size() - pad_top; i < output.size(); i++)
            std::fill(output[i].begin(), output[i].end(), 0.0f);
        // Left and right columns for the middle rows
        for (int i = pad_top; i < pad_top + rows; i++) {
            std::fill(output[i].begin(), output[i].begin()+pad_left, 0.0f);
            std::fill(output[i].end()- (totalPadCols - pad_left), output[i].end(), 0.0f);
        }
    }
    else if (padding == Padding::CONSTANT) {
        for (int i = 0; i < pad_top; i++)
            std::fill(output[i].begin(), output[i].end(), pad_val);
        for (int i = output.size() - pad_top; i < output.size(); i++)
            std::fill(output[i].begin(), output[i].end(), pad_val);
        for (int i = pad_top; i < pad_top + rows; i++) {
            std::fill(output[i].begin(), output[i].begin()+pad_left, pad_val);
            std::fill(output[i].end()- (totalPadCols - pad_left), output[i].end(), pad_val);
        }
    }
    else if (padding == Padding::REPLICATE) {
        // Top rows: replicate first row of input
        for (int i = 0; i < pad_top; i++) {
            output[i] = output[pad_top]; // copy entire row
        }
        // Bottom rows: replicate last row of input
        for (int i = output.size() - pad_top; i < output.size(); i++) {
            output[i] = output[pad_top + rows - 1];
        }
        // Left and right columns: replicate first and last column of each row
        for (int i = pad_top; i < pad_top + rows; i++) {
            for (int j = 0; j < pad_left; j++)
                output[i][j] = output[i][pad_left];
            for (int j = output[0].size() - (totalPadCols - pad_left); j < output[0].size(); j++)
                output[i][j] = output[i][pad_left + cols - 1];
        }
    }
    else if (padding == Padding::REFLECT) {
        // Top: reflect rows from below the border
        for (int i = 0; i < pad_top; i++) {
            output[i] = output[pad_top + (pad_top - i)];
        }
        // Bottom: reflect from above the bottom border
        for (int i = 0; i < pad_top; i++) {
            output[output.size()-1-i] = output[output.size()-pad_top-1-i];
        }
        // Left/right: reflect within each row
        for (int i = pad_top; i < pad_top + rows; i++) {
            for (int j = 0; j < pad_left; j++) {
                output[i][j] = output[i][pad_left*2 - j];
            }
            for (int j = 0; j < pad_left; j++) {
                output[i][output[0].size()-1-j] = output[i][output[0].size()-1-pad_left*2+j];
            }
        }
    }
    else {
        throw std::invalid_argument("Unsupported padding type");
    }
}

// -------------------- 1D Convolution via FFT --------------------
// This function computes the full convolution (length = L1 + L2 - 1)
// using FFT. The caller is responsible for cropping the result.
static void convolve1D(const std::vector<float>& input_signal, 
                       const std::vector<float>& kernel,
                       std::vector<float>& full_conv,
                       FFTWrapper& fftWrapper) {
    int L1 = input_signal.size();
    int L2 = kernel.size();
    int fullSize = L1 + L2 - 1;
    
    // Zero-pad both input and kernel to fullSize.
    std::vector<float> padded_signal(fullSize, 0.0f);
    std::vector<float> padded_kernel(fullSize, 0.0f);
    std::copy(input_signal.begin(), input_signal.end(), padded_signal.begin());
    std::copy(kernel.begin(), kernel.end(), padded_kernel.begin());
    
    // The FFT (real-to-complex) output size is fullSize/2+1.
    int fftSize = fullSize / 2 + 1;
    std::vector<std::complex<float>> fft_signal(fftSize);
    std::vector<std::complex<float>> fft_kernel(fftSize);
    std::vector<std::complex<float>> fft_product(fftSize);
    
    if (fftWrapper.performFFT1D(padded_signal, fft_signal, Flags::ESTIMATE) != FFTStatus::SUCCESS)
        throw std::runtime_error("FFT1D failed");
    if (fftWrapper.performFFT1D(padded_kernel, fft_kernel, Flags::ESTIMATE) != FFTStatus::SUCCESS)
        throw std::runtime_error("FFT1D failed");
    
    for (int i = 0; i < fftSize; i++) {
        fft_product[i] = fft_signal[i] * fft_kernel[i];
    }
    
    full_conv.resize(fullSize);
    if (fftWrapper.performInverseFFT1D(fft_product, full_conv, Flags::ESTIMATE) != FFTStatus::SUCCESS)
        throw std::runtime_error("Inverse FFT1D failed");
}

// -------------------- Overloaded 1D Convolve --------------------
// This function calls the FFT convolution and then crops the result
// to either “same” (output length equals input length) or “valid”.
std::vector<float> convolve(const std::vector<float>& signal,
                            const std::vector<float>& kernel,
                            int stride, // not used
                            Padding padding,
                            float pad_val) {
    // Determine how to pad the input before convolution.
    int inputSize = signal.size();
    int kernelSize = kernel.size();
    std::vector<float> signalToConv;
    
    if (padding != Padding::VALID) {
        // For “same” convolution we extend the signal.
        int paddedSize = inputSize + kernelSize - 1;
        signalToConv.resize(paddedSize, 0.0f);
        pad1D(signal, signalToConv, padding, pad_val);
    } else {
        // VALID: no extra padding.
        signalToConv = signal;
    }
    
    // Compute full convolution (length = paddedSize + kernelSize - 1).
    int paddedSize = signalToConv.size();
    std::vector<float> full_conv;
    FFTWrapper fftWrapper;
    convolve1D(signalToConv, kernel, full_conv, fftWrapper);
    
    // Crop the result.
    std::vector<float> output;
    if (padding == Padding::VALID) {
        // VALID: output length = inputSize - kernelSize + 1.
        int outSize = inputSize - kernelSize + 1;
        // For VALID the “full” result (when no padding was applied)
        // is of length = inputSize + kernelSize - 1 so we take indices [kernelSize-1, kernelSize-1+outSize].
        int start = kernelSize - 1;
        output.assign(full_conv.begin() + start, full_conv.begin() + start + outSize);
    } else {
        // SAME: output length = inputSize.
        int outSize = inputSize;
        int start = (full_conv.size() - outSize) / 2;
        output.assign(full_conv.begin() + start, full_conv.begin() + start + outSize);
    }
    
    return output;
}

// -------------------- 2D Convolution via FFT --------------------
// This helper computes the full 2D convolution between two matrices.
// (The kernel is flipped inside to compute the convolution.)
static void convolve2D(const std::vector<std::vector<float>>& input, 
                       const std::vector<std::vector<float>>& kernel,
                       std::vector<std::vector<float>>& full_conv,
                       FFTWrapper& fftWrapper) {
    int inRows = input.size();
    int inCols = input[0].size();
    int kRows = kernel.size();
    int kCols = kernel[0].size();
    int fullRows = inRows + kRows - 1;
    int fullCols = inCols + kCols - 1;
    
    // Zero-pad input into a full-sized array.
    std::vector<std::vector<float>> padded_input(fullRows, std::vector<float>(fullCols, 0.0f));
    for (int i = 0; i < inRows; i++)
        for (int j = 0; j < inCols; j++)
            padded_input[i][j] = input[i][j];
    
    // Create a padded kernel by flipping it.
    std::vector<std::vector<float>> padded_kernel(fullRows, std::vector<float>(fullCols, 0.0f));
    for (int i = 0; i < kRows; i++)
        for (int j = 0; j < kCols; j++)
            padded_kernel[i][j] = kernel[kRows - i - 1][kCols - j - 1];
    
    // Perform 2D FFT on both arrays.
    std::vector<std::vector<std::complex<float>>> fft_input;
    std::vector<std::vector<std::complex<float>>> fft_kernel;
    if (fftWrapper.performFFT2D(padded_input, fft_input, Flags::ESTIMATE) != FFTStatus::SUCCESS)
        throw std::runtime_error("FFT2D failed");
    if (fftWrapper.performFFT2D(padded_kernel, fft_kernel, Flags::ESTIMATE) != FFTStatus::SUCCESS)
        throw std::runtime_error("FFT2D failed");
    
    int fftCols = fft_input[0].size(); // equals fullCols/2+1
    std::vector<std::vector<std::complex<float>>> fft_product(fullRows, std::vector<std::complex<float>>(fftCols));
    for (int i = 0; i < fullRows; i++) {
        for (int j = 0; j < fftCols; j++) {
            fft_product[i][j] = fft_input[i][j] * fft_kernel[i][j];
        }
    }
    
    if (fftWrapper.performInverseFFT2D(fft_product, full_conv, Flags::ESTIMATE) != FFTStatus::SUCCESS)
        throw std::runtime_error("Inverse FFT2D failed");
}

// -------------------- Overloaded 2D Convolve --------------------
// This function pads the input image if needed, calls the FFT convolution,
// and then crops the “full convolution” to produce either “same” or “valid” output.
std::vector<std::vector<float>> convolve(const std::vector<std::vector<float>>& image,
                                         const std::vector<std::vector<float>>& kernel,
                                         int stride, // not used
                                         Padding padding,
                                         float pad_val) {
    int imgRows = image.size();
    int imgCols = image[0].size();
    int kRows = kernel.size();
    int kCols = kernel[0].size();
    
    std::vector<std::vector<float>> inputToConv;
    if (padding != Padding::VALID) {
        // For same padding, pad the image so that the output is the same size.
        int paddedRows = imgRows + kRows - 1;
        int paddedCols = imgCols + kCols - 1;
        inputToConv.resize(paddedRows, std::vector<float>(paddedCols, 0.0f));
        pad2D(image, inputToConv, padding, pad_val);
    }
    else {
        inputToConv = image;
    }
    
    // Compute the full 2D convolution.
    std::vector<std::vector<float>> full_conv;
    FFTWrapper fftWrapper;
    convolve2D(inputToConv, kernel, full_conv, fftWrapper);
    
    std::vector<std::vector<float>> output;
    if (padding == Padding::VALID) {
        // VALID: output size = (imgRows - kRows + 1) x (imgCols - kCols + 1)
        int outRows = imgRows - kRows + 1;
        int outCols = imgCols - kCols + 1;
        output.resize(outRows, std::vector<float>(outCols, 0.0f));
        int startRow = kRows - 1;
        int startCol = kCols - 1;
        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                output[i][j] = full_conv[i + startRow][j + startCol];
            }
        }
    }
    else {
        // SAME: output size = original image size.
        int outRows = imgRows;
        int outCols = imgCols;
        output.resize(outRows, std::vector<float>(outCols, 0.0f));
        int startRow = (full_conv.size() - outRows) / 2;
        int startCol = (full_conv[0].size() - outCols) / 2;
        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                output[i][j] = full_conv[i + startRow][j + startCol];
            }
        }
    }
    
    return output;
}

// -------------------- Utility Functions --------------------
void print1DVector(const std::vector<float>& vec) {
    for (const auto& val : vec)
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    std::cout << std::endl;
}

void print2DVector(const std::vector<std::vector<float>>& mat) {
    for (const auto& row : mat) {
        for (const auto& val : row)
            std::cout << std::fixed << std::setprecision(2) << val << " ";
        std::cout << std::endl;
    }
}

// -------------------- Main Test --------------------
int main() {
    // Test 1D Convolution
    std::vector<float> signal = {1, 2, 3, 4};
    std::vector<float> kernel = {1, 0, -1};
    
    std::cout << "1D Convolution Test:" << std::endl;
    std::vector<float> output_signal;
    
    output_signal = convolve(signal, kernel, 1, Padding::ZERO);
    std::cout << "Output with ZERO padding:" << std::endl;
    print1DVector(output_signal);
    
    output_signal = convolve(signal, kernel, 1, Padding::CONSTANT, 2.0f);
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
    std::vector<std::vector<float>> output_image;
    
    output_image = convolve(image, kernel2D, 1, Padding::ZERO);
    std::cout << "Output with ZERO padding:" << std::endl;
    print2DVector(output_image);
    
    output_image = convolve(image, kernel2D, 1, Padding::CONSTANT, 2.0f);
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
