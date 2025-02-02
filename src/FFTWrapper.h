// /src/FFTWrapper.h

#ifndef FFTWRAPPER_H
#define FFTWRAPPER_H

#include <vector>
#include <complex>
#include <fftw3.h>

// Flags for FFTW plans.
enum class Flags {
    ESTIMATE = FFTW_ESTIMATE,
    MEASURE  = FFTW_MEASURE
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
    
    // 1D FFT (real-to-complex)
    FFTStatus performFFT1D(const std::vector<float>& input,
                           std::vector<std::complex<float>>& output,
                           Flags flags);
    
    // 1D Inverse FFT (complex-to-real)
    FFTStatus performInverseFFT1D(const std::vector<std::complex<float>>& input,
                                  std::vector<float>& output,
                                  Flags flags);
    
    // 2D FFT (real-to-complex)
    FFTStatus performFFT2D(const std::vector<std::vector<float>>& input,
                           std::vector<std::vector<std::complex<float>>>& output,
                           Flags flags);
    
    // 2D Inverse FFT (complex-to-real)
    FFTStatus performInverseFFT2D(const std::vector<std::vector<std::complex<float>>>& input,
                                  std::vector<std::vector<float>>& output,
                                  Flags flags);
};

#endif // FFTWRAPPER_H
