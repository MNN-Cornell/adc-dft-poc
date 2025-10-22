import numpy as np
import matplotlib.pyplot as plt

ADC_RESOLUTION = 8
ADC_MAX_VALUE = 2**ADC_RESOLUTION - 1
ADC_SAMPLING_RATE = 500_000/10000

def generate_adc_signal(frequencies_dict, duration_periods=10, noise_level=0.0):
    """
    Generate a simulated ADC output signal from multiple frequencies.
    
    Args:
        frequencies_dict: Dict with format {freq_hz: weight, ...}
                         Weights should sum to 1.0
                         Example: {1000: 0.5, 2000: 0.5}
        duration_periods: Number of periods to generate (default 10)
        noise_level: Standard deviation of Gaussian noise (0-1 scale, relative to ADC range)
    
    Returns:
        samples: Array of ADC values (0 to ADC_MAX_VALUE)
        sample_times: Corresponding time values
    """
    min_freq = min(frequencies_dict.keys())
    period = 1 / min_freq
    total_duration = period * duration_periods
    
    num_samples = int(ADC_SAMPLING_RATE * total_duration)
    sample_times = np.linspace(0, total_duration, num_samples, endpoint=False)
    
    signal = np.zeros_like(sample_times)
    for freq, weight in frequencies_dict.items():
        signal += weight * np.sin(2 * np.pi * freq * sample_times)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(signal))
        signal = signal + noise
    
    signal = np.clip(signal, -1, 1)
    
    adc_samples = ((signal + 1) / 2 * ADC_MAX_VALUE).astype(np.uint16)
    
    return adc_samples, sample_times


def compute_dft(samples, apply_window=True):
    """
    Compute the Discrete Fourier Transform of the signal.
    
    Args:
        samples: Array of ADC samples
        apply_window: If True, applies Hamming window before FFT
    
    Returns:
        frequencies: Frequency bins (Hz)
        magnitude_spectrum: Magnitude of each frequency component
        phase_spectrum: Phase of each frequency component
    """
    N = len(samples)
    
    if apply_window:
        window = np.hamming(N)
        windowed_samples = samples * window
    else:
        windowed_samples = samples
    
    dft = np.fft.fft(windowed_samples)
    
    frequencies = np.fft.fftfreq(N, d=1/ADC_SAMPLING_RATE)
    
    magnitude_spectrum = np.abs(dft) / N
    phase_spectrum = np.angle(dft)
    
    return frequencies, magnitude_spectrum, phase_spectrum


def plot_spectrum(frequencies, magnitude_spectrum, title="ADC DFT Spectrum"):
    """
    Plot the frequency spectrum from DFT results.
    
    Args:
        frequencies: Frequency bins (Hz)
        magnitude_spectrum: Magnitude of each frequency component
        title: Title for the plot
    """
    positive_freq_idx = frequencies >= 0
    freqs_positive = frequencies[positive_freq_idx]
    mag_positive = magnitude_spectrum[positive_freq_idx]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs_positive, mag_positive, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_original_signal(samples, sample_times, title="Original ADC Signal"):
    """
    Plot the original ADC signal.
    
    Args:
        samples: Array of ADC samples
        sample_times: Corresponding time values
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample_times * 1e6, samples, linewidth=0.8)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("ADC Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_all_analysis(samples, times, freq_bins, magnitude, frequencies_dict):
    """
    Create comprehensive plots with original signal, bit representation, overall DFT spectrum,
    and individual bit stream DFT spectra in a separate window.
    Automatically saves both figures.

    Args:
        samples: Array of ADC samples
        times: Corresponding time values
        freq_bins: Frequency bins from overall DFT
        magnitude: Magnitude spectrum from overall DFT
        frequencies_dict: Dictionary of input frequencies
    """
    # --- First Window: Original Signal, Bit Representation, Overall DFT ---
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Original ADC Signal
    axes1[0].plot(times * 1e6, samples, linewidth=0.8)
    axes1[0].set_xlabel("Time (µs)")
    axes1[0].set_ylabel("ADC Value")
    axes1[0].set_title(f"Original ADC Signal - {list(frequencies_dict.keys())} Hz")
    axes1[0].grid(True, alpha=0.3)

    # Plot 2: 8-Bit Digital Representation
    bit_matrix = np.array([(samples >> bit) & 1 for bit in range(ADC_RESOLUTION)])
    for bit in range(ADC_RESOLUTION):
        bit_line = bit_matrix[bit]
        y_offset = ADC_RESOLUTION - 1 - bit
        axes1[1].step(times * 1e6, bit_line + y_offset, where='post', linewidth=2, label=f"Bit {bit}")
    axes1[1].set_xlabel("Time (µs)")
    axes1[1].set_ylabel("Bit")
    axes1[1].set_title(f"{ADC_RESOLUTION}-Bit Digital Representation")
    axes1[1].set_ylim(-0.5, ADC_RESOLUTION - 0.5)
    axes1[1].set_yticks(range(ADC_RESOLUTION))
    axes1[1].set_yticklabels([f"Bit {i}" for i in range(ADC_RESOLUTION)])
    axes1[1].set_xlim(axes1[0].get_xlim())
    axes1[1].grid(True, alpha=0.3, axis='x')
    axes1[1].set_facecolor('#f5f5f5')

    # Plot 3: Overall ADC DFT Spectrum
    positive_freq_idx = freq_bins >= 0
    freqs_positive = freq_bins[positive_freq_idx]
    mag_positive = magnitude[positive_freq_idx]
    axes1[2].stem(freqs_positive, mag_positive, basefmt=' ')
    axes1[2].set_xlabel("Frequency (Hz)")
    axes1[2].set_ylabel("Magnitude")
    axes1[2].set_title(f"Overall ADC DFT Spectrum")
    axes1[2].grid(True, alpha=0.3, axis='y')

    fig1.tight_layout() # Apply tight_layout to the first figure
    fig1.savefig('figure1_analysis.png') # Save the first figure

    # Store the x-axis limits for DFT plots to align them in the second window
    dft_xlim = axes1[2].get_xlim()

    # --- Second Window: Individual Bit Stream DFTs ---
    fig2, axes2 = plt.subplots(ADC_RESOLUTION, 1, figsize=(14, 4 * ADC_RESOLUTION))
    fig2.suptitle("Individual Bit Stream DFT Spectra", fontsize=16) # Add a suptitle for the second window

    # Ensure axes2 is always an array, even if ADC_RESOLUTION is 1
    if ADC_RESOLUTION == 1:
        axes2 = [axes2]

    # Plot DFT for each bit stream
    for bit in range(ADC_RESOLUTION):
        bit_stream = bit_matrix[bit]
        bit_freq_bins, bit_magnitude, _ = compute_dft(bit_stream, apply_window=False)

        positive_bit_freq_idx = bit_freq_bins >= 0
        bit_freqs_positive = bit_freq_bins[positive_bit_freq_idx]
        bit_mag_positive = bit_magnitude[positive_bit_freq_idx]

        axes2[bit].stem(bit_freqs_positive, bit_mag_positive, basefmt=' ')
        axes2[bit].set_xlabel("Frequency (Hz)")
        axes2[bit].set_ylabel("Magnitude")
        axes2[bit].set_title(f"DFT Spectrum for Bit {bit}")
        axes2[bit].grid(True, alpha=0.3, axis='y')
        axes2[bit].set_xlim(dft_xlim) # Align x-axis with the overall DFT

    fig2.tight_layout() # Apply tight_layout to the second figure
    plt.subplots_adjust(top=0.95) # Adjust top to make space for suptitle
    fig2.savefig('figure2_bit_dfts.png') # Save the second figure

    plt.show() # Show both figures


if __name__ == "__main__":
    frequencies = {2: 0.5, 5: 0.5}
    
    samples, times = generate_adc_signal(frequencies_dict=frequencies, duration_periods=10, noise_level=0)
    
    print(f"Generated {len(samples)} samples")
    print(f"Sample rate: {ADC_SAMPLING_RATE} Hz")
    print(f"Signal frequencies: {frequencies}")
    print(f"ADC value range: 0 to {ADC_MAX_VALUE}")
    print(f"Sample min: {samples.min()}, max: {samples.max()}")
    
    freq_bins, magnitude, phase = compute_dft(samples, apply_window=False)
    
    plot_all_analysis(samples, times, freq_bins, magnitude, frequencies)
