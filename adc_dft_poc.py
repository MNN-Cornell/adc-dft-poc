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
    Create a comprehensive plot with original signal, bit representation, and DFT spectrum.
    
    Args:
        samples: Array of ADC samples
        times: Corresponding time values
        freq_bins: Frequency bins from DFT
        magnitude: Magnitude spectrum from DFT
        frequencies_dict: Dictionary of input frequencies
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    axes[0].plot(times * 1e6, samples, linewidth=0.8)
    axes[0].set_xlabel("Time (µs)")
    axes[0].set_ylabel("ADC Value")
    axes[0].set_title(f"Original ADC Signal - {list(frequencies_dict.keys())} Hz")
    axes[0].grid(True, alpha=0.3)
    
    bit_matrix = np.array([(samples >> bit) & 1 for bit in range(ADC_RESOLUTION)])
    
    for bit in range(ADC_RESOLUTION):
        bit_line = bit_matrix[bit]
        y_offset = ADC_RESOLUTION - 1 - bit
        axes[1].step(times * 1e6, bit_line + y_offset, where='post', linewidth=2, label=f"Bit {bit}")
    
    axes[1].set_xlabel("Time (µs)")
    axes[1].set_ylabel("Bit")
    axes[1].set_title(f"{ADC_RESOLUTION}-Bit Digital Representation")
    axes[1].set_ylim(-0.5, ADC_RESOLUTION - 0.5)
    axes[1].set_yticks(range(ADC_RESOLUTION))
    axes[1].set_yticklabels([f"Bit {i}" for i in range(ADC_RESOLUTION)])
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_facecolor('#f5f5f5')
    
    positive_freq_idx = freq_bins >= 0
    freqs_positive = freq_bins[positive_freq_idx]
    mag_positive = magnitude[positive_freq_idx]
    axes[2].stem(freqs_positive, mag_positive, basefmt=' ')
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude")
    axes[2].set_title(f"ADC DFT Spectrum")
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


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
