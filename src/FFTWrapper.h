#ifndef FFTWRAPPER_H
#define FFTWRAPPER_H

#include <fftw3.h>

enum class Flags {
    FFTW_ESTIMATE = FFTW_ESTIMATE,  // quick but less optimal planning
    FFTW_MEASURE = FFTW_MEASURE     // slower but optimal planning
};

enum class FFTStatus {
    SUCCESS,
    FAILURE
};

class FFTWrapper {
public:
    FFTWrapper();
    ~FFTWrapper();

    // 1D FFT Functions
    FFTStatus performFFT1D(float* input, fftwf_complex* output, int size, Flags flags);
    FFTStatus performInverseFFT1D(fftwf_complex* input, float* output, int size, Flags flags);

    // 2D FFT Functions
    FFTStatus performFFT2D(float* input, fftwf_complex* output, int rows, int cols, Flags flags);
    FFTStatus performInverseFFT2D(fftwf_complex* input, float* output, int rows, int cols, Flags flags);

private:
    fftwf_plan forwardPlan1D;
    fftwf_plan inversePlan1D;
    fftwf_plan forwardPlan2D;
    fftwf_plan inversePlan2D;
};

#endif // FFTWRAPPER_H