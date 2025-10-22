# ADC DFT Proof of Concept

A simulation tool for analyzing Raspberry Pi Pico ADC signals using Discrete Fourier Transform (DFT).

## Features

- **Multi-frequency signal generation** - Combine multiple sine waves with adjustable weights
- **ADC simulation** - 8-bit or 12-bit resolution (configurable)
- **Binary visualization** - View individual bit transitions over time
- **DFT analysis** - Compute frequency spectrum with Hamming window
- **Comprehensive plots** - Original signal, bit representation, and frequency spectrum

## Usage

```python
python adc_dft_poc.py
```

## Configuration

Edit the `__main__` section to customize:

```python
frequencies = {2: 0.5, 5: 0.5}  # Frequencies (Hz) and their weights
samples, times = generate_adc_signal(
    frequencies_dict=frequencies,
    duration_periods=10,
    noise_level=0
)
```

## Parameters

- **ADC_RESOLUTION**: 8 or 12 bits
- **ADC_SAMPLING_RATE**: Sampling frequency in Hz
- **frequencies**: Dict of {frequency_hz: weight}
- **duration_periods**: Number of signal periods to generate
- **noise_level**: Gaussian noise standard deviation (0-1)

## Output

Two separate figure windows are displayed, and both are automatically saved as PNG files:

**Figure 1: `figure1_analysis.png`**
This window contains three plots:
1. **Original ADC Signal** - Analog waveform over time
2. **8-Bit Digital Representation** - Individual bit transitions
3. **Overall ADC DFT Spectrum** - Frequency components of the full ADC signal (stem plot)

**Figure 2: `figure2_bit_dfts.png`**
This window contains individual DFT spectra for each bit stream (from Bit 0 to Bit 7 for an 8-bit ADC). Each plot shows the frequency components of a single bit's digital stream.
