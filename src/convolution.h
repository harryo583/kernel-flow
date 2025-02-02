#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

// Padding type used by the convolution routines.
enum class Padding {
    VALID,      // No padding.
    ZERO,       // Pad with zeros.
    CONSTANT,   // Pad with a constant value.
    REPLICATE,  // Replicate edge values.
    REFLECT     // Reflect border values.
};

// 1D convolution function (using FFT).
// The 'stride' parameter is accepted but not implemented.
std::vector<float> convolve(const std::vector<float>& signal,
                            const std::vector<float>& kernel,
                            int stride,
                            Padding padding,
                            float pad_val = 0.0f);

// 2D convolution function (using FFT).
// The 'stride' parameter is accepted but not implemented.
std::vector<std::vector<float>> convolve(const std::vector<std::vector<float>>& image,
                                         const std::vector<std::vector<float>>& kernel,
                                         int stride,
                                         Padding padding,
                                         float pad_val = 0.0f);

// Utility functions to print 1D and 2D vectors.
void print1DVector(const std::vector<float>& vec);
void print2DVector(const std::vector<std::vector<float>>& mat);

#endif // CONVOLUTION_H
