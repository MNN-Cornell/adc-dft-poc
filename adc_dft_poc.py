import numpy as np
import matplotlib.pyplot as plt

# --- ADC & Signal Configuration ---

# ADC resolution in bits. This determines the number of discrete digital levels.
# An 8-bit ADC has 2^8 = 256 levels.
ADC_RESOLUTION = 8

# Maximum digital value for the ADC. For an 8-bit ADC, this is 255.
ADC_MAX_VALUE = 2**ADC_RESOLUTION - 1

# Sampling rate of the ADC in Hz.
# This is the number of samples taken per second.
# The value 500_000 / 10000 is a simplified representation of a potential real-world scenario
# where a high-speed clock is divided down for the ADC.
ADC_SAMPLING_RATE = 500_000 / 10000

def generate_adc_signal(frequencies_dict, duration_periods=10, noise_level=0.0):
    """
    Generates a simulated ADC output signal from a combination of sine waves.

    This function creates a composite signal from multiple frequency components,
    adds optional noise, and then simulates the quantization process of an ADC.

    Args:
        frequencies_dict (dict): A dictionary where keys are frequencies in Hz
                                 and values are their corresponding weights in the signal.
                                 The weights should ideally sum to 1.0 for a normalized signal.
                                 Example: {1000: 0.5, 2000: 0.5}
        duration_periods (int): The number of periods of the lowest frequency to generate.
                                This determines the total duration of the signal. (Default: 10)
        noise_level (float): The standard deviation of Gaussian noise to add to the signal.
                             This is on a 0-1 scale relative to the ADC's input range. (Default: 0.0)

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of quantized ADC sample values (0 to ADC_MAX_VALUE).
            - np.ndarray: An array of corresponding time values for each sample.
    """
    # Determine the lowest frequency to calculate the signal's fundamental period.
    min_freq = min(frequencies_dict.keys())
    period = 1 / min_freq
    total_duration = period * duration_periods

    # Calculate the total number of samples based on duration and sampling rate.
    num_samples = int(ADC_SAMPLING_RATE * total_duration)
    # Generate a time vector for the samples.
    sample_times = np.linspace(0, total_duration, num_samples, endpoint=False)

    # Create the composite signal by summing weighted sine waves.
    signal = np.zeros_like(sample_times)
    for freq, weight in frequencies_dict.items():
        signal += weight * np.sin(2 * np.pi * freq * sample_times)

    # Add Gaussian noise if specified.
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(signal))
        signal += noise

    # Clip the signal to the range [-1, 1] to simulate the ADC's input voltage limits.
    signal = np.clip(signal, -1, 1)

    # --- ADC Quantization ---
    # 1. Scale the signal from [-1, 1] to [0, 2].
    # 2. Normalize to the ADC's range [0, ADC_MAX_VALUE].
    # 3. Convert to an integer type to represent the digital output.
    adc_samples = ((signal + 1) / 2 * ADC_MAX_VALUE).astype(np.uint16)

    return adc_samples, sample_times


def compute_dft(samples, apply_window=True):
    """
    Computes the Discrete Fourier Transform (DFT) of a signal.

    This function can apply a windowing function to reduce spectral leakage
    before performing the Fast Fourier Transform (FFT).

    Args:
        samples (np.ndarray): An array of ADC samples.
        apply_window (bool): If True, applies a Hamming window to the samples
                             before the FFT. (Default: True)

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of frequency bins in Hz.
            - np.ndarray: The magnitude spectrum of the signal.
            - np.ndarray: The phase spectrum of the signal.
    """
    N = len(samples)

    # Apply a Hamming window to reduce spectral leakage.
    if apply_window:
        window = np.hamming(N)
        windowed_samples = samples * window
    else:
        windowed_samples = samples

    # Compute the FFT.
    dft = np.fft.fft(windowed_samples)

    # Calculate the frequency bins for the DFT.
    frequencies = np.fft.fftfreq(N, d=1 / ADC_SAMPLING_RATE)

    # --- Spectrum Calculation ---
    # 1. Calculate the magnitude and normalize by the number of samples (N).
    # 2. Calculate the phase angle of each frequency component.
    magnitude_spectrum = np.abs(dft) / N
    phase_spectrum = np.angle(dft)

    return frequencies, magnitude_spectrum, phase_spectrum


def plot_spectrum(frequencies, magnitude_spectrum, title="ADC DFT Spectrum"):
    """
    Plots the frequency spectrum from DFT results.

    This function visualizes the magnitude of each frequency component.
    It only plots the positive frequency components, as the negative
    frequencies are a mirror image for real-valued signals.

    Args:
        frequencies (np.ndarray): An array of frequency bins in Hz.
        magnitude_spectrum (np.ndarray): The magnitude of each frequency component.
        title (str): The title for the plot. (Default: "ADC DFT Spectrum")

    Returns:
        tuple: A tuple containing the Matplotlib figure and axes objects.
    """
    # Filter for positive frequencies only.
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
    Plots the original time-domain ADC signal.

    Args:
        samples (np.ndarray): An array of ADC samples.
        sample_times (np.ndarray): An array of corresponding time values.
        title (str): The title for the plot. (Default: "Original ADC Signal")

    Returns:
        tuple: A tuple containing the Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot time in microseconds for better readability.
    ax.plot(sample_times * 1e6, samples, linewidth=0.8)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("ADC Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_all_analysis(samples, times, freq_bins, magnitude, frequencies_dict):
    """
    Creates and saves a comprehensive set of plots for analysis.

    This function generates two separate figures:
    1.  A figure with the original signal, its digital bit representation,
        and the overall DFT spectrum.
    2.  A figure showing the individual DFT spectra for each bit stream.

    Both figures are automatically saved to PNG files.

    Args:
        samples (np.ndarray): An array of ADC samples.
        times (np.ndarray): An array of corresponding time values.
        freq_bins (np.ndarray): An array of frequency bins from the overall DFT.
        magnitude (np.ndarray): The magnitude spectrum from the overall DFT.
        frequencies_dict (dict): The dictionary of input frequencies for labeling.
    """
    # --- Figure 1: Original Signal, Bit Representation, and Overall DFT ---
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Original ADC Signal
    axes1[0].plot(times * 1e6, samples, linewidth=0.8)
    axes1[0].set_xlabel("Time (µs)")
    axes1[0].set_ylabel("ADC Value")
    axes1[0].set_title(f"Original ADC Signal - {list(frequencies_dict.keys())} Hz")
    axes1[0].grid(True, alpha=0.3)

    # Plot 2: Digital Bit Representation
    # Extract each bit from the ADC samples into a matrix.
    bit_matrix = np.array([(samples >> bit) & 1 for bit in range(ADC_RESOLUTION)])
    for bit in range(ADC_RESOLUTION):
        bit_line = bit_matrix[bit]
        # Offset each bit line vertically for clear visualization.
        y_offset = ADC_RESOLUTION - 1 - bit
        axes1[1].step(times * 1e6, bit_line + y_offset, where='post', linewidth=2, label=f"Bit {bit}")
    axes1[1].set_xlabel("Time (µs)")
    axes1[1].set_ylabel("Bit")
    axes1[1].set_title(f"{ADC_RESOLUTION}-Bit Digital Representation")
    axes1[1].set_ylim(-0.5, ADC_RESOLUTION - 0.5)
    axes1[1].set_yticks(range(ADC_RESOLUTION))
    axes1[1].set_yticklabels([f"Bit {i}" for i in range(ADC_RESOLUTION)])
    axes1[1].set_xlim(axes1[0].get_xlim()) # Align x-axis with the signal plot.
    axes1[1].grid(True, alpha=0.3, axis='x')
    axes1[1].set_facecolor('#f5f5f5') # Light gray background for clarity.

    # Plot 3: Overall ADC DFT Spectrum
    positive_freq_idx = freq_bins > 0
    freqs_positive = freq_bins[positive_freq_idx]
    mag_positive = magnitude[positive_freq_idx]
    axes1[2].stem(freqs_positive, mag_positive, basefmt=' ')
    axes1[2].set_xlabel("Frequency (Hz)")
    axes1[2].set_ylabel("Magnitude")
    axes1[2].set_title("Overall ADC DFT Spectrum")
    axes1[2].grid(True, alpha=0.3, axis='y')

    fig1.tight_layout()
    fig1.savefig('figure1_analysis.png') # Save the first figure.

    # Store the x-axis limits to align the DFT plots in the second figure.
    dft_xlim = axes1[2].get_xlim()

    # --- Figure 2: Individual Bit Stream DFTs ---
    fig2, axes2 = plt.subplots(ADC_RESOLUTION, 1, figsize=(14, 4 * ADC_RESOLUTION))
    fig2.suptitle("Individual Bit Stream DFT Spectra", fontsize=16)

    # Ensure axes2 is always an array for consistent indexing.
    if ADC_RESOLUTION == 1:
        axes2 = [axes2]

    # Compute and plot the DFT for each individual bit stream.
    for bit in range(ADC_RESOLUTION):
        bit_stream = bit_matrix[bit]
        # Compute DFT for the single bit stream.
        bit_freq_bins, bit_magnitude, _ = compute_dft(bit_stream, apply_window=False)

        positive_bit_freq_idx = bit_freq_bins > 0
        bit_freqs_positive = bit_freq_bins[positive_bit_freq_idx]
        bit_mag_positive = bit_magnitude[positive_bit_freq_idx]

        axes2[bit].stem(bit_freqs_positive, bit_mag_positive, basefmt=' ')
        axes2[bit].set_xlabel("Frequency (Hz)")
        axes2[bit].set_ylabel("Magnitude")
        axes2[bit].set_title(f"DFT Spectrum for Bit {bit}")
        axes2[bit].grid(True, alpha=0.3, axis='y')
        axes2[bit].set_xlim(dft_xlim) # Align x-axis with the overall DFT plot.

    fig2.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust layout to make space for the suptitle.
    fig2.savefig('figure2_bit_dfts.png') # Save the second figure.

    plt.show() # Display both figures.


if __name__ == "__main__":
    # --- Simulation Parameters ---
    # Define the frequencies (in Hz) and their relative weights for the input signal.
    frequencies = {2: 0.5, 5: 0.5}

    # Generate the ADC signal based on the specified parameters.
    samples, times = generate_adc_signal(
        frequencies_dict=frequencies,
        duration_periods=10,  # Generate 10 periods of the lowest frequency.
        noise_level=0         # No noise.
    )

    # --- Output Information ---
    print(f"Generated {len(samples)} samples")
    print(f"Sample rate: {ADC_SAMPLING_RATE} Hz")
    print(f"Signal frequencies: {frequencies}")
    print(f"ADC value range: 0 to {ADC_MAX_VALUE}")
    print(f"Sample min: {samples.min()}, max: {samples.max()}")

    # Compute the DFT of the generated ADC signal.
    # A windowing function is not applied here to maintain peak clarity for this specific simulation.
    freq_bins, magnitude, phase = compute_dft(samples, apply_window=False)

    # Generate and display all analysis plots.
    plot_all_analysis(samples, times, freq_bins, magnitude, frequencies)