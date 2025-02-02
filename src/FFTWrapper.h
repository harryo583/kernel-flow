#ifndef FFTWRAPPER_H
#define FFTWRAPPER_H

#include <vector>
#include <complex>
#include <fftw3.h>

// FFTW planning flags.
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
    
    // Perform a 1D real-to-complex FFT.
    FFTStatus performFFT1D(const std::vector<float>& input,
                           std::vector<std::complex<float>>& output,
                           Flags flags);
    
    // Perform a 1D complex-to-real inverse FFT.
    FFTStatus performInverseFFT1D(const std::vector<std::complex<float>>& input,
                                  std::vector<float>& output,
                                  Flags flags);
    
    // Perform a 2D real-to-complex FFT.
    FFTStatus performFFT2D(const std::vector<std::vector<float>>& input,
                           std::vector<std::vector<std::complex<float>>>& output,
                           Flags flags);
    
    // Perform a 2D complex-to-real inverse FFT.
    FFTStatus performInverseFFT2D(const std::vector<std::vector<std::complex<float>>>& input,
                                  std::vector<std::vector<float>>& output,
                                  Flags flags);
};

#endif // FFTWRAPPER_H
