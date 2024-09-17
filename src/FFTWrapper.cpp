#include "FFTWrapper.h"
#include <fftw3.h>
#include <iostream>

enum class Flags {
    FFTW_ESTIMATE = FFTW_ESTIMATE,  // quick but less optimal planning
    FFTW_MEASURE = FFTW_MEASURE     // slower but optimal planning
};

enum class FFTStatus {
    SUCCESS,
    FAILURE
};

// Constructor
FFTWrapper::FFTWrapper() {
}

// Destructor
FFTWrapper::~FFTWrapper() {
    fftwf_cleanup();
}

// ------------------- 1D FFT Implementation -------------------

// Function to perform the forward FFT (Real-to-Complex)
FFTStatus FFTWrapper::performFFT1D(float* input, fftwf_complex* output, int size, Flags flags) {
    fftwf_plan forwardPlan1D = fftwf_plan_dft_r2c_1d(size, input, output, static_cast<unsigned int>(flags));
    if (forwardPlan1D == nullptr) {
        return FFTStatus::FAILURE;
    }
    
    fftwf_execute(forwardPlan1D);
    fftwf_destroy_plan(forwardPlan1D);
    return FFTStatus::SUCCESS;
}

// Function to perform the inverse FFT (Complex-to-Real)
FFTStatus FFTWrapper::performInverseFFT1D(fftwf_complex* input, float* output, int size, Flags flags) {
    fftwf_plan inversePlan1D = fftwf_plan_dft_c2r_1d(size, input, output, static_cast<unsigned int>(flags));
    if (inversePlan1D == nullptr) {
        return FFTStatus::FAILURE;
    }
    
    fftwf_execute(inversePlan1D);

    // Normalize the output
    for (int i = 0; i < size; i++) {
        output[i] /= size;
    }
    fftwf_destroy_plan(inversePlan1D);
    return FFTStatus::SUCCESS;
}

// ------------------- 2D FFT Implementation -------------------

// Function to perform the forward FFT (Real-to-Complex)
FFTStatus FFTWrapper::performFFT2D(float* input, fftwf_complex* output, int rows, int cols, Flags flags) {
    fftwf_plan forwardPlan2D = fftwf_plan_dft_r2c_2d(rows, cols, input, output, static_cast<unsigned int>(flags));
    if (forwardPlan2D == nullptr) {
        return FFTStatus::FAILURE;
    }
    
    fftwf_execute(forwardPlan2D);
    fftwf_destroy_plan(forwardPlan2D);
    return FFTStatus::SUCCESS;
}

// Function to perform the inverse FFT (Complex-to-Real)
FFTStatus FFTWrapper::performInverseFFT2D(fftwf_complex* input, float* output, int rows, int cols, Flags flags) {
    fftwf_plan inversePlan2D = fftwf_plan_dft_c2r_2d(rows, cols, input, output, static_cast<unsigned int>(flags));
    if (inversePlan2D == nullptr) {
        return FFTStatus::FAILURE;
    }

    fftwf_execute(inversePlan2D);

    // Normalize the output
    for (int i = 0; i < rows * cols; i++) {
        output[i] /= (rows * cols);
    }
    fftwf_destroy_plan(inversePlan2D);
    return FFTStatus::SUCCESS;
}