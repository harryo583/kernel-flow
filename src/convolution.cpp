
#include <iostream>
#include <vector>
#include "FFTWrapper.h"

enum class Padding {
    ZERO, // pads the image or feature map with zeros
    CONSTANT, // pads with a user-specified constant
    SAME, // ensures that the output size is the same as the input size
    REPLICATE, // pads by replicating the edge pixels of the image or feature map
    REFLECT, // pads by reflecting border pixels
}

static void pad1D(const std::vector<float>& input, std::vector<float>& output, Padding padding) {
    int len = input.size(); // length of input
    int klen = output.size() - len; // length of kernel

    if (padding == Padding::ZERO) {
        return;
    } else if (padding == Padding::REPLICATE) {
        // TODO:
        return;
    } else if (padding == Padding::REFLECT) {
        // TODO:
        return;
    } else {
        throw std::invalid_argument("Unsupported padding type"); 
    }
}

static void pad2D(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, Padding padding) {
    int rows = input.size(); // height of input image
    int cols = input[0].size(); // width of input image
    int krows = output.size() - rows; // height of kernel
    int kcols = output[0].size() - cols; // width of kernel

    if (padding == Padding::ZERO) {
        return;
    } else if (padding == Padding::REPLICATE) {
        // TODO: parallelize
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[krows/2 + i][kcols/2 + j] = input[i][j];
            }
        }
        // Replicate edges
        for (int j = 0; j < output[0].size(); ++j) {
            output[0][j] = output[krows/2][j];
            output[output.size() - 1][j] = output[output.size() - krows/2 - 1][j];
        }
        for (int i = 0; i < output.size(); ++i) {
            output[i][0] = output[i][kcols/2];
            output[i][output[0].size() - 1] = output[i][output[0].size() - kcols/2 - 1];
        }
        return;
    } else if (padding == Padding::REFLECT) {
        // TODO: parallelize
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[krows/2 + i][kcols/2 + j] = input[i][j];
            }
        }
        // Reflect edges
        for (int j = 0; j < output[0].size(); ++j) {
            output[0][j] = output[1][j];
            output[output.size() - 1][j] = output[output.size() - 2][j];
        }
        for (int i = 0; i < output.size(); ++i) {
            output[i][0] = output[i][1];
            output[i][output[0].size() - 1] = output[i][output[0].size() - 2];
        }
    } else {
        throw std::invalid_argument("Unsupported padding type"); 
    }
}

static void convolve1D()
{
    return;
}

static void convolve2D(const std::vector<std::vector<float>>& padded_image,
              const std::vector<std::vector<float>>& kernel,
              std::vector<std::vector<float>>& output_image,
              FFTWrapper& fftWrapper)
{
    return;
}

std::vector<std::vector<float>> convolve(const std::vector<std::vector<float>>& image, const std::vector<std::vector<float>>& kernel, int stride = 1, Padding padding = Padding::ZERO) {
    return;
}