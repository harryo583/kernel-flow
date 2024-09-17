#ifndef FFTWRAPPER_H
#define FFTWRAPPER_H

#include <fftw3.h>
#include <vector>
#include <complex>

enum class Flags {
    ESTIMATE = FFTW_ESTIMATE,
    MEASURE = FFTW_MEASURE
};

enum class FFTStatus {
    SUCCESS,
    FAILURE
};

class FFTWrapper {
private:
    fftwf_plan forwardPlan1D;
    fftwf_plan inversePlan1D;
    fftwf_plan forwardPlan2D;
    fftwf_plan inversePlan2D;

public:
    FFTWrapper();
    ~FFTWrapper();

    // Perform 1D FFT (Real to Complex)
    FFTStatus performFFT1D(const std::vector<float>& input, std::vector<std::complex<float>>& output, Flags flags);

    // Perform 1D Inverse FFT (Complex to Real)
    FFTStatus performInverseFFT1D(const std::vector<std::complex<float>>& input, std::vector<float>& output, Flags flags);

    // Perform 2D FFT (Real to Complex)
    FFTStatus performFFT2D(const std::vector<std::vector<float>>& input, std::vector<std::vector<std::complex<float>>>& output, Flags flags);

    // Perform 2D Inverse FFT (Complex to Real)
    FFTStatus performInverseFFT2D(const std::vector<std::vector<std::complex<float>>>& input, std::vector<std::vector<float>>& output, Flags flags);
};

#endif  // FFTWRAPPER_H
