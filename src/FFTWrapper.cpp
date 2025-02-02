// /src/FFTWrapper.cpp

#include "FFTWrapper.h"
#include <fftw3.h>
#include <stdexcept>
#include <iostream>

// Constructor
FFTWrapper::FFTWrapper() {
    forwardPlan1D = nullptr;
    inversePlan1D = nullptr;
    forwardPlan2D = nullptr;
    inversePlan2D = nullptr;
}

// Destructor: destroy any remaining plans.
FFTWrapper::~FFTWrapper() {
    if (forwardPlan1D) fftwf_destroy_plan(forwardPlan1D);
    if (inversePlan1D) fftwf_destroy_plan(inversePlan1D);
    if (forwardPlan2D) fftwf_destroy_plan(forwardPlan2D);
    if (inversePlan2D) fftwf_destroy_plan(inversePlan2D);
}

// ------------------- 1D FFT -------------------
FFTStatus FFTWrapper::performFFT1D(const std::vector<float>& input,
                                   std::vector<std::complex<float>>& output,
                                   Flags flags) {
    int size = input.size();
    output.resize(size / 2 + 1);
    forwardPlan1D = fftwf_plan_dft_r2c_1d(size, const_cast<float*>(input.data()),
                                          reinterpret_cast<fftwf_complex*>(output.data()),
                                          static_cast<unsigned int>(flags));
    if (!forwardPlan1D)
        return FFTStatus::FAILURE;
    fftwf_execute(forwardPlan1D);
    fftwf_destroy_plan(forwardPlan1D);
    forwardPlan1D = nullptr;
    return FFTStatus::SUCCESS;
}

FFTStatus FFTWrapper::performInverseFFT1D(const std::vector<std::complex<float>>& input,
                                          std::vector<float>& output,
                                          Flags flags) {
    int size = output.size();
    inversePlan1D = fftwf_plan_dft_c2r_1d(size,
                                          reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(input.data())),
                                          output.data(),
                                          static_cast<unsigned int>(flags));
    if (!inversePlan1D)
        return FFTStatus::FAILURE;
    fftwf_execute(inversePlan1D);
    fftwf_destroy_plan(inversePlan1D);
    inversePlan1D = nullptr;
    for (auto& val : output)
        val /= size;
    return FFTStatus::SUCCESS;
}

// ------------------- 2D FFT -------------------
FFTStatus FFTWrapper::performFFT2D(const std::vector<std::vector<float>>& input,
                                   std::vector<std::vector<std::complex<float>>>& output,
                                   Flags flags) {
    int rows = input.size();
    int cols = input[0].size();
    output.resize(rows, std::vector<std::complex<float>>(cols / 2 + 1));
    std::vector<float> flatInput(rows * cols);
    std::vector<std::complex<float>> flatOutput(rows * (cols / 2 + 1));
    
    // Flatten the 2D input.
    for (int i = 0; i < rows; ++i)
        std::copy(input[i].begin(), input[i].end(), flatInput.begin() + i * cols);
    
    forwardPlan2D = fftwf_plan_dft_r2c_2d(rows, cols, flatInput.data(),
                                          reinterpret_cast<fftwf_complex*>(flatOutput.data()),
                                          static_cast<unsigned int>(flags));
    if (!forwardPlan2D)
        return FFTStatus::FAILURE;
    fftwf_execute(forwardPlan2D);
    fftwf_destroy_plan(forwardPlan2D);
    forwardPlan2D = nullptr;
    
    // Un-flatten into 2D.
    int fftCols = cols / 2 + 1;
    for (int i = 0; i < rows; ++i)
        std::copy(flatOutput.begin() + i * fftCols,
                  flatOutput.begin() + (i + 1) * fftCols,
                  output[i].begin());
    return FFTStatus::SUCCESS;
}

FFTStatus FFTWrapper::performInverseFFT2D(const std::vector<std::vector<std::complex<float>>>& input,
                                          std::vector<std::vector<float>>& output,
                                          Flags flags) {
    int rows = input.size();
    int fftCols = input[0].size();
    int realCols = (fftCols - 1) * 2;  // original number of columns
    output.resize(rows, std::vector<float>(realCols));
    
    std::vector<std::complex<float>> flatInput(rows * fftCols);
    std::vector<float> flatOutput(rows * realCols);
    
    for (int i = 0; i < rows; ++i)
        std::copy(input[i].begin(), input[i].end(), flatInput.begin() + i * fftCols);
    
    inversePlan2D = fftwf_plan_dft_c2r_2d(rows, realCols,
                                          reinterpret_cast<fftwf_complex*>(flatInput.data()),
                                          flatOutput.data(),
                                          static_cast<unsigned int>(flags));
    if (!inversePlan2D)
        return FFTStatus::FAILURE;
    fftwf_execute(inversePlan2D);
    fftwf_destroy_plan(inversePlan2D);
    inversePlan2D = nullptr;
    
    for (int i = 0; i < rows; ++i)
        std::copy(flatOutput.begin() + i * realCols,
                  flatOutput.begin() + (i + 1) * realCols,
                  output[i].begin());
    
    // Normalize the output.
    for (auto& row : output)
        for (auto& val : row)
            val /= (rows * realCols);
    
    return FFTStatus::SUCCESS;
}
