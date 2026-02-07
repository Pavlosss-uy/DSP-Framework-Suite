# DSP Framework Suite
A comprehensive Python-based Digital Signal Processing Suite featuring a Tkinter GUI for real-time signal generation, transformation, and filtering.

## Key Features
* **Signal Operations:** Supports basic operations including addition, subtraction, shifting, folding (reversing), and linear quantization.
* **Frequency Domain Analysis:** Direct and Inverse Fourier Transforms implemented from scratch to analyze signal magnitude and phase.
* **Advanced Correlation:** Features for direct correlation, time-delay estimation, and signal classification based on average max correlation.
* **FIR Filter Designer:** Automated design of Low Pass, High Pass, Band Pass, and Band Stop filters using various windowing methods (Rectangular, Hanning, Hamming, Blackman).

## 🛠 Technical Implementation
### Windowing Logic
The framework automatically selects the window type ($w(n)$) and calculates the filter order ($N$) based on the Stopband Attenuation ($\delta_s$) provided by the user:
- **Rectangular:** $\delta_s \leq 21$ dB
- **Hanning:** $\delta_s \leq 44$ dB
- **Hamming:** $\delta_s \leq 53$ dB
- **Blackman:** $\delta_s \leq 74$ dB

### Signal Processing Pipeline
- **Filtering:** Applied via time-domain convolution using the `convolute` function.
- **Normalization:** Frequencies are normalized by the sampling frequency ($F_s$) and adjusted using half-transition bands to suit the window method.

## Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the application: `python main.py`.
