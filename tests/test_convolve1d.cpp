#include <iostream>
#include <vector>
#include <cassert>
#include "src/FFTWrapper.h"

void printVector(const std::vector<float>& vec) {
    for (float val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void testConvolve1D() {
    FFTWrapper fftWrapper;
    
    // Test 1: Zero Padding
    {
        std::vector<float> signal = {1, 2, 3};
        std::vector<float> kernel = {0, 1, 0};
        int stride = 1;
        Padding padding = Padding::ZERO;

        std::vector<float> expected = {0, 1, 2, 3, 0};
        std::vector<float> result = convolve(signal, kernel, stride, padding);

        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); ++i) {
            assert(abs(result[i] - expected[i]) < 1e-6); // Allow small floating-point errors
        }
    }

    // Test 2: Constant Padding
    {
        std::vector<float> signal = {1, 2};
        std::vector<float> kernel = {1, 0, 0, 1};
        int stride = 1;
        Padding padding = Padding::CONSTANT;

        std::vector<float> expected = {1, 2, 0, 0, 0, 1};
        std::vector<float> result = convolve(signal, kernel, stride, padding, 0);

        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); ++i) {
            assert(abs(result[i] - expected[i]) < 1e-6);
        }
    }
}

int main() {
    testConvolve1D();
    std::cout << "All 1D convolution tests passed!" << std::endl;
    return 0;
}
