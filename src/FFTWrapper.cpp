#include "FFTWrapper.h"
#include <fftw3.h>
#include <iostream>

// Constructor
FFTWrapper::FFTWrapper() {
    forwardPlan1D = nullptr;
    inversePlan1D = nullptr;
    forwardPlan2D = nullptr;
    inversePlan2D = nullptr;
}

// Destructor
FFTWrapper::~FFTWrapper() {
    if (forwardPlan1D) fftwf_destroy_plan(forwardPlan1D);
    if (inversePlan1D) fftwf_destroy_plan(inversePlan1D);
    if (forwardPlan2D) fftwf_destroy_plan(forwardPlan2D);
    if (inversePlan2D) fftwf_destroy_plan(inversePlan2D);
}

// ------------------- 1D FFT Implementation -------------------

FFTStatus FFTWrapper::performFFT1D(const std::vector<float>& input, std::vector<std::complex<float>>& output, Flags flags) {
    int size = input.size();
    output.resize(size / 2 + 1);

    forwardPlan1D = fftwf_plan_dft_r2c_1d(size, const_cast<float*>(input.data()),
                                          reinterpret_cast<fftwf_complex*>(output.data()), static_cast<unsigned int>(flags));

    if (!forwardPlan1D) {
        return FFTStatus::FAILURE;
    }

    fftwf_execute(forwardPlan1D);
    return FFTStatus::SUCCESS;
}

FFTStatus FFTWrapper::performInverseFFT1D(const std::vector<std::complex<float>>& input, std::vector<float>& output, Flags flags) {
    int size = output.size();

    inversePlan1D = fftwf_plan_dft_c2r_1d(size, reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(input.data())), 
                                          output.data(), static_cast<unsigned int>(flags));

    if (!inversePlan1D) {
        return FFTStatus::FAILURE;
    }

    fftwf_execute(inversePlan1D);

    // Normalize the output
    for (auto& val : output) {
        val /= size;
    }

    return FFTStatus::SUCCESS;
}

// ------------------- 2D FFT Implementation -------------------

FFTStatus FFTWrapper::performFFT2D(const std::vector<std::vector<float>>& input, std::vector<std::vector<std::complex<float>>>& output, Flags flags) {
    int rows = input.size();
    int cols = input[0].size();

    output.resize(rows, std::vector<std::complex<float>>(cols / 2 + 1)); // Resize output for complex values

    // Create an intermediate 1D array for FFTW
    std::vector<float> flatInput(rows * cols);
    std::vector<std::complex<float>> flatOutput(rows * (cols / 2 + 1));

    // Flatten the 2D input array to 1D
    for (int i = 0; i < rows; ++i) {
        std::copy(input[i].begin(), input[i].end(), flatInput.begin() + i * cols);
    }

    // Create FFTW plans
    forwardPlan2D = fftwf_plan_dft_r2c_2d(rows, cols, flatInput.data(),
                                          reinterpret_cast<fftwf_complex*>(flatOutput.data()), static_cast<unsigned int>(flags));

    if (!forwardPlan2D) {
        return FFTStatus::FAILURE;
    }

    // Execute FFT
    fftwf_execute(forwardPlan2D);

    // Copy the results to the 2D output
    for (int i = 0; i < rows; ++i) {
        std::copy(flatOutput.begin() + i * (cols / 2 + 1), flatOutput.begin() + (i + 1) * (cols / 2 + 1), output[i].begin());
    }

    return FFTStatus::SUCCESS;
}

FFTStatus FFTWrapper::performInverseFFT2D(const std::vector<std::vector<std::complex<float>>>& input, std::vector<std::vector<float>>& output, Flags flags) {
    int rows = input.size();
    int cols = input[0].size();

    output.resize(rows, std::vector<float>(cols * 2)); // Resize output for real values

    // Create an intermediate 1D array for FFTW
    std::vector<std::complex<float>> flatInput(rows * (cols / 2 + 1));
    std::vector<float> flatOutput(rows * cols * 2);

    // Flatten the 2D input array to 1D
    for (int i = 0; i < rows; ++i) {
        std::copy(input[i].begin(), input[i].end(), flatInput.begin() + i * (cols / 2 + 1));
    }

    // Create FFTW plans
    inversePlan2D = fftwf_plan_dft_c2r_2d(rows, cols * 2, reinterpret_cast<fftwf_complex*>(flatInput.data()),
                                          flatOutput.data(), static_cast<unsigned int>(flags));

    if (!inversePlan2D) {
        return FFTStatus::FAILURE;
    }

    // Execute inverse FFT
    fftwf_execute(inversePlan2D);

    // Copy the results to the 2D output
    for (int i = 0; i < rows; ++i) {
        std::copy(flatOutput.begin() + i * (cols * 2), flatOutput.begin() + (i + 1) * (cols * 2), output[i].begin());
    }

    // Normalize the output
    for (auto& row : output) {
        for (auto& val : row) {
            val /= (rows * cols);
        }
    }

    return FFTStatus::SUCCESS;
}