#include <vector>
#include <iostream>
#include <stdexcept>
#include <omp.h>

enum class Padding {
    ZERO, // pads the image or feature map with zeros
    CONSTANT, // pads with a user-specified constant value
    SAME, // ensures that the output size is the same as the input size
    REPLICATE, // pads by replicating the edge pixels of the image or feature map
    REFLECT, // pads by reflecting border pixels
}

class Convolution {
public:
    Convolution(const std::vector<std::vector<float>>& kernel, int stride = 1, Padding padding = Padding::ZERO)
    : kernel_(kernel), stride_(stride), padding_(padding) {
        krows_ = kernel_.size();
        kcols_ = kernel_[0].size();
    }

    std::vector<std::vector<float>> convolve(const std::vector<std::vector<float>>& image) {
        int rows = image.size();
        int cols = image[0].size();

        // Compute the padded image dimensions
        int padded_rows;
        int padded_cols;

        if (padding == Padding::SAME) {
            padded_rows = rows;
            padded_cols = cols;
        } else if (padding == Padding::CONSTANT || padding == Padding::REFLECT || padding == Padding::REPLICATE || padding == Padding::ZERO) {
            padded_rows = rows + (krows_ - 1) / 2;
            padded_cols = cols + (kcols_ - 1) / 2;
        } else {
            throw std::invalid_argument("Unsupported padding type");
        }

        // Create empty padded image
        std::vector<std::vector<float>> padded_image(padded_rows, std::vector<float>(padded_cols, 0));

        // TODO: implement padding based on padding type (ZERO, CONSTANT, REPLICATE, REFLECT)
        

        // Create the output image
        int output_rows = (padded_rows - krows_) / stride_ + 1;
        int output_cols = (padded_cols - kcols_) / stride_ + 1;
        std::vector<std::vector<float>> output_image(output_rows, std::vector<float>(output_cols, 0));

        // TODO: implement the convolution operation

        return output_image
    }

private:
    std::vector<std::vector<float>> kernel_;
    int stride_; // stride
    Padding padding; // padding type
    int krows_; // number of rows in the kernel
    int kcols_; // number of columns in the kernel
};