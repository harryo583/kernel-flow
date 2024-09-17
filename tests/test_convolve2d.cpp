#include <iostream>
#include <vector>
#include <cassert>
#include "src/FFTWrapper.h"

void print2DVector(const std::vector<std::vector<float>>& vec) {
    for (const auto& row : vec) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void testConvolve2D() {
    FFTWrapper fftWrapper;
    
    // Test 1: Zero Padding
    {
        std::vector<std::vector<float>> image = {{1, 2}, {3, 4}};
        std::vector<std::vector<float>> kernel = {{0, 1}, {1, 0}};
        int stride = 1;
        Padding padding = Padding::ZERO;

        std::vector<std::vector<float>> expected = {
            {0, 1, 1},
            {1, 2, 1},
            {1, 1, 0}
        };

        std::vector<std::vector<float>> result = convolve(image, kernel, stride, padding);

        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); ++i) {
            assert(result[i].size() == expected[i].size());
            for (size_t j = 0; j < result[i].size(); ++j) {
                assert(abs(result[i][j] - expected[i][j]) < 1e-6); // Allow small floating-point errors
            }
        }
    }

    // Test 2: Constant Padding
    {
        std::vector<std::vector<float>> image = {{1, 2}, {3, 4}};
        std::vector<std::vector<float>> kernel = {{1, 0}, {0, 1}};
        int stride = 1;
        Padding padding = Padding::CONSTANT;

        std::vector<std::vector<float>> expected = {
            {1, 2, 1},
            {3, 4, 3},
            {1, 1, 0}
        };

        std::vector<std::vector<float>> result = convolve(image, kernel, stride, padding, 0);

        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); ++i) {
            assert(result[i].size() == expected[i].size());
            for (size_t j = 0; j < result[i].size(); ++j) {
                assert(abs(result[i][j] - expected[i][j]) < 1e-6);
            }
        }
    }
}

int main() {
    testConvolve2D();
    std::cout << "All 2D convolution tests passed!" << std::endl;
    return 0;
}
