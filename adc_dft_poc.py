# adc_dft_poc.py
#
# Author: Gemini
# Date: 2025-10-23
#
# Description:
# This script simulates the process of an Analog-to-Digital Converter (ADC)
# and performs a Discrete Fourier Transform (DFT) on the resulting digital signal.
# It is designed to help visualize and understand the frequency components
# present in a quantized signal and its individual bitstreams.
#
# The script generates a composite signal from multiple sine waves, simulates
# ADC quantization, and then computes and plots the DFT of the overall signal
# as well as the DFT of each individual bit of the ADC output. This is useful
# for analyzing how different frequency components of the analog signal
# manifest in the digital domain.

import numpy as np
import matplotlib.pyplot as plt

# --- ADC & Signal Configuration ---

# ADC_RESOLUTION: ADC resolution in bits.
# This determines the number of discrete digital levels the ADC can represent.
# An 8-bit ADC has 2^8 = 256 levels, ranging from 0 to 255.
ADC_RESOLUTION = 8

# ADC_MAX_VALUE: The maximum digital value for the ADC.
# For an N-bit ADC, this is 2^N - 1.
ADC_MAX_VALUE = 2**ADC_RESOLUTION - 1

# ADC_SAMPLING_RATE: The sampling rate of the ADC in Hz.
# This is the number of samples taken per second from the analog signal.
# The Nyquist-Shannon sampling theorem states that the sampling rate must be at
# least twice the highest frequency component in the signal to avoid aliasing.
# The value 500_000 / 10000 is a simplified representation of a potential
# real-world scenario where a high-speed clock is divided down for the ADC.
ADC_SAMPLING_RATE = 500_000 / 10000

def generate_adc_signal(frequencies_dict, duration_periods=10, noise_level=0.0):
    """
    Generates a simulated ADC output signal from a combination of sine waves.

    This function creates a composite analog signal from multiple frequency
    components, adds optional Gaussian noise, and then simulates the quantization
    process of an ADC to produce a digital signal.

    Args:
        frequencies_dict (dict): A dictionary where keys are frequencies in Hz
                                 and values are their corresponding weights
                                 (amplitudes) in the signal. The weights should
                                 ideally sum to 1.0 for a normalized signal to
                                 fit within the ADC's [-1, 1] input range.
                                 Example: {1000: 0.5, 2000: 0.5}
        duration_periods (int): The number of periods of the lowest frequency
                                to generate. This determines the total duration
                                of the signal and affects the frequency
                                resolution of the subsequent DFT. (Default: 10)
        noise_level (float): The standard deviation of Gaussian noise to add to
                             the analog signal. This is on a 0-1 scale relative
                             to the ADC's input range. (Default: 0.0)

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of quantized ADC sample values (integers from
                          0 to ADC_MAX_VALUE).
            - np.ndarray: An array of corresponding time values for each sample.
    """
    # Determine the lowest frequency to calculate the signal's fundamental period.
    # This is used to set the overall duration of the generated signal.
    min_freq = min(frequencies_dict.keys())
    period = 1 / min_freq
    total_duration = period * duration_periods

    # Calculate the total number of samples to generate based on the signal
    # duration and the ADC's sampling rate.
    num_samples = int(ADC_SAMPLING_RATE * total_duration)
    # Generate a time vector representing the time at each sample point.
    sample_times = np.linspace(0, total_duration, num_samples, endpoint=False)

    # Create the composite analog signal by summing the weighted sine waves.
    signal = np.zeros_like(sample_times)
    for freq, weight in frequencies_dict.items():
        signal += weight * np.sin(2 * np.pi * freq * sample_times)

    # Add Gaussian (white) noise to the signal if a noise level is specified.
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(signal))
        signal += noise

    # Clip the signal to the range [-1, 1] to simulate the ADC's input
    # voltage limits. Any part of the signal outside this range is saturated.
    signal = np.clip(signal, -1, 1)

    # --- ADC Quantization ---
    # This process converts the continuous analog signal into a discrete digital signal.
    # 1. Scale the signal from its [-1, 1] range to [0, 2].
    # 2. Normalize this to the ADC's digital range [0, ADC_MAX_VALUE].
    # 3. Convert the floating-point values to integers, representing the final
    #    digital output of the ADC.
    adc_samples = ((signal + 1) / 2 * ADC_MAX_VALUE).astype(np.uint16)

    return adc_samples, sample_times


def compute_dft(samples, apply_window=True):
    """
    Computes the Discrete Fourier Transform (DFT) of a signal.

    This function transforms a time-domain signal into the frequency domain.
    It can apply a windowing function to the signal before the transform to
    reduce spectral leakage, which occurs when the signal is not periodic
    within the sampling window.

    Args:
        samples (np.ndarray): An array of ADC samples (the time-domain signal).
        apply_window (bool): If True, applies a Hamming window to the samples
                             before performing the FFT. This is generally
                             recommended for non-periodic signals. (Default: True)

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of frequency bins in Hz.
            - np.ndarray: The magnitude spectrum of the signal.
            - np.ndarray: The phase spectrum of the signal.
    """
    N = len(samples)

    # Apply a window function to the samples to reduce spectral leakage.
    # The Hamming window is a common choice that tapers the signal at its
    # beginning and end.
    if apply_window:
        window = np.hamming(N)
        windowed_samples = samples * window
    else:
        windowed_samples = samples

    # Compute the Fast Fourier Transform (FFT), an efficient algorithm for DFT.
    dft = np.fft.fft(windowed_samples)

    # Calculate the frequency bins corresponding to each point in the DFT output.
    frequencies = np.fft.fftfreq(N, d=1 / ADC_SAMPLING_RATE)

    # --- Spectrum Calculation ---
    # 1. Calculate the magnitude of the complex DFT output. This represents the
    #    amplitude of each frequency component. It's normalized by the number
    #    of samples (N) to make it independent of signal length.
    # 2. Calculate the phase angle of each frequency component. This represents
    #    the phase shift of the component.
    magnitude_spectrum = np.abs(dft) / N
    phase_spectrum = np.angle(dft)

    return frequencies, magnitude_spectrum, phase_spectrum


def plot_spectrum(frequencies, magnitude_spectrum, title="ADC DFT Spectrum"):
    """
    Plots the frequency spectrum from DFT results.

    This function visualizes the magnitude of each frequency component. For
    real-valued input signals, the DFT is symmetric, so this function only
    plots the positive frequency components.

    Args:
        frequencies (np.ndarray): An array of frequency bins in Hz from the DFT.
        magnitude_spectrum (np.ndarray): The magnitude of each frequency component.
        title (str): The title for the plot. (Default: "ADC DFT Spectrum")

    Returns:
        tuple: A tuple containing the Matplotlib figure and axes objects.
    """
    # Filter for positive frequencies only, as the negative frequency part of
    # the spectrum is a mirror image for real signals and adds no new information.
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
    # Plot time in microseconds (µs) for better readability on the x-axis.
    ax.plot(sample_times * 1e6, samples, linewidth=0.8)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("ADC Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_all_analysis(samples, times, freq_bins, magnitude, frequencies_dict):
    """
    Creates and saves a comprehensive set of plots for analysis.

    This function generates four separate figures for a detailed analysis:
    1.  A figure showing the original signal, its digital bit representation
        over time, and the overall DFT spectrum of the quantized signal in Hz,
        angular frequency, and normalized frequency.
    2.  A figure showing the individual DFT spectra for each bit stream of the
        ADC output (frequency in Hz).
    3.  A figure showing the individual DFT spectra for each bit stream with
        angular frequency (rad/s) on x-axis.
    4.  A figure showing the individual DFT spectra for each bit stream with
        normalized frequency on x-axis.

    All figures are automatically saved to PNG files.

    Args:
        samples (np.ndarray): An array of ADC samples.
        times (np.ndarray): An array of corresponding time values.
        freq_bins (np.ndarray): An array of frequency bins from the overall DFT.
        magnitude (np.ndarray): The magnitude spectrum from the overall DFT.
        frequencies_dict (dict): The dictionary of input frequencies for labeling.
    """
    # --- Figure 1: Original Signal, Bit Representation, and Overall DFT (Hz, Angular, Normalized) ---
    fig1, axes1 = plt.subplots(5, 1, figsize=(14, 20))

    # Plot 1: Original ADC Signal
    axes1[0].plot(times * 1e6, samples, linewidth=0.8)
    axes1[0].set_xlabel("Time (µs)")
    axes1[0].set_ylabel("ADC Value")
    axes1[0].set_title(f"Original ADC Signal - Frequencies: {list(frequencies_dict.keys())} Hz")
    axes1[0].grid(True, alpha=0.3)

    # Plot 2: Digital Bit Representation
    # Extract each bit from the ADC samples into a matrix. Each row of the
    # matrix corresponds to a bit (from LSB to MSB).
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
    axes1[1].set_facecolor('#f5f5f5') # Light gray background for better readability.

    # Plot 3: Overall ADC DFT Spectrum (Hz) - Exclude DC component (f=0)
    positive_freq_idx = freq_bins > 0
    freqs_positive = freq_bins[positive_freq_idx]
    mag_positive = magnitude[positive_freq_idx]
    axes1[2].stem(freqs_positive, mag_positive, basefmt=' ')
    axes1[2].set_xlabel("Frequency (Hz)")
    axes1[2].set_ylabel("Magnitude")
    axes1[2].set_title("Overall ADC DFT Spectrum (Frequency in Hz)")
    axes1[2].grid(True, alpha=0.3, axis='y')

    # Plot 4: Overall ADC DFT Spectrum (Angular Frequency) - Exclude DC component
    # Angular frequency: ω = 2πf, range is 0 to 2π×fs (not limited to 2π!)
    all_freq_idx = freq_bins > 0  # Exclude f=0
    freqs_all = freq_bins[all_freq_idx]
    mag_all = magnitude[all_freq_idx]
    
    angular_freqs_all = 2 * np.pi * freqs_all
    angular_freqs_in_pi = angular_freqs_all / np.pi
    
    axes1[3].stem(angular_freqs_in_pi, mag_all, basefmt=' ')
    axes1[3].set_xlabel("Angular Frequency (×π rad/s)")
    axes1[3].set_ylabel("Magnitude")
    axes1[3].set_title("Overall ADC DFT Spectrum (Angular Frequency)")
    axes1[3].grid(True, alpha=0.3, axis='y')

    # Plot 5: Overall ADC DFT Spectrum (Normalized Frequency) - Full spectrum 0 to 2π
    normalized_freqs_all = 2 * np.pi * freqs_all / ADC_SAMPLING_RATE
    # Map negative frequencies to the [π, 2π] range
    normalized_freqs_all = np.where(normalized_freqs_all < 0, normalized_freqs_all + 2 * np.pi, normalized_freqs_all)
    normalized_freqs_in_pi = normalized_freqs_all / np.pi
    
    # Sort by normalized frequency for proper display
    sort_idx_norm = np.argsort(normalized_freqs_in_pi)
    axes1[4].stem(normalized_freqs_in_pi[sort_idx_norm], mag_all[sort_idx_norm], basefmt=' ')
    axes1[4].set_xlabel("Normalized Frequency (×π rad/sample)")
    axes1[4].set_ylabel("Magnitude")
    axes1[4].set_title("Overall ADC DFT Spectrum (Normalized Frequency)")
    axes1[4].grid(True, alpha=0.3, axis='y')
    axes1[4].set_xlim([0, 2])

    fig1.tight_layout()
    fig1.savefig('figure1_analysis.png') # Save the first figure.

    # Store the x-axis limits to align the DFT plots in the subsequent figures.
    dft_xlim = axes1[2].get_xlim()
    angular_xlim = axes1[3].get_xlim()  # This is already in units of π
    normalized_xlim = axes1[4].get_xlim()

    # --- Figure 2: Individual Bit Stream DFTs ---
    fig2, axes2 = plt.subplots(ADC_RESOLUTION, 1, figsize=(14, 4 * ADC_RESOLUTION))
    fig2.suptitle("Individual Bit Stream DFT Spectra", fontsize=16)

    # Ensure axes2 is always an array for consistent indexing, even if ADC_RESOLUTION is 1.
    if ADC_RESOLUTION == 1:
        axes2 = [axes2]

    # Compute and plot the DFT for each individual bit stream (exclude DC).
    for bit in range(ADC_RESOLUTION):
        bit_stream = bit_matrix[bit]
        # Compute DFT for the single bit stream. A windowing function is not
        # applied here to see the raw spectral content of the digital signal.
        bit_freq_bins, bit_magnitude, _ = compute_dft(bit_stream, apply_window=False)

        positive_bit_freq_idx = bit_freq_bins > 0  # Exclude f=0
        bit_freqs_positive = bit_freq_bins[positive_bit_freq_idx]
        bit_mag_positive = bit_magnitude[positive_bit_freq_idx]

        axes2[bit].stem(bit_freqs_positive, bit_mag_positive, basefmt=' ')
        axes2[bit].set_xlabel("Frequency (Hz)")
        axes2[bit].set_ylabel("Magnitude")
        axes2[bit].set_title(f"DFT Spectrum for Bit {bit}")
        axes2[bit].grid(True, alpha=0.3, axis='y')
        axes2[bit].set_xlim(dft_xlim) # Align x-axis with the overall DFT plot for comparison.

    fig2.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust layout to make space for the suptitle.
    fig2.savefig('figure2_bit_dfts.png') # Save the second figure.

    # --- Figure 3: Individual Bit Stream DFTs (Angular Frequency) ---
    fig3, axes3 = plt.subplots(ADC_RESOLUTION, 1, figsize=(14, 4 * ADC_RESOLUTION))
    fig3.suptitle("Individual Bit Stream DFT Spectra (Angular Frequency)", fontsize=16)

    # Ensure axes3 is always an array for consistent indexing, even if ADC_RESOLUTION is 1.
    if ADC_RESOLUTION == 1:
        axes3 = [axes3]

    # Compute and plot the DFT for each individual bit stream (angular frequency, exclude DC).
    for bit in range(ADC_RESOLUTION):
        bit_stream = bit_matrix[bit]
        bit_freq_bins, bit_magnitude, _ = compute_dft(bit_stream, apply_window=False)

        # Exclude DC component (f=0)
        all_bit_freq_idx = bit_freq_bins > 0
        bit_freqs_all = bit_freq_bins[all_bit_freq_idx]
        bit_mag_all = bit_magnitude[all_bit_freq_idx]
        
        bit_angular_freqs_all = 2 * np.pi * bit_freqs_all
        bit_angular_freqs_in_pi = bit_angular_freqs_all / np.pi
        
        axes3[bit].stem(bit_angular_freqs_in_pi, bit_mag_all, basefmt=' ')
        axes3[bit].set_xlabel("Angular Frequency (×π rad/s)")
        axes3[bit].set_ylabel("Magnitude")
        axes3[bit].set_title(f"DFT Spectrum for Bit {bit} (Angular Frequency)")
        axes3[bit].grid(True, alpha=0.3, axis='y')
        axes3[bit].set_xlim(angular_xlim) # Align x-axis with the overall DFT plot for comparison.

    fig3.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust layout to make space for the suptitle.
    fig3.savefig('figure3_bit_dfts_angular.png') # Save the third figure.

    # --- Figure 4: Individual Bit Stream DFTs (Normalized Frequency) ---
    fig4, axes4 = plt.subplots(ADC_RESOLUTION, 1, figsize=(14, 4 * ADC_RESOLUTION))
    fig4.suptitle("Individual Bit Stream DFT Spectra (Normalized Frequency)", fontsize=16)

    # Ensure axes4 is always an array for consistent indexing, even if ADC_RESOLUTION is 1.
    if ADC_RESOLUTION == 1:
        axes4 = [axes4]

    # Compute and plot the DFT for each individual bit stream (normalized frequency, exclude DC).
    for bit in range(ADC_RESOLUTION):
        bit_stream = bit_matrix[bit]
        bit_freq_bins, bit_magnitude, _ = compute_dft(bit_stream, apply_window=False)

        # Exclude DC component (f=0)
        all_bit_freq_idx = bit_freq_bins > 0
        bit_freqs_all = bit_freq_bins[all_bit_freq_idx]
        bit_mag_all = bit_magnitude[all_bit_freq_idx]
        
        bit_normalized_freqs_all = 2 * np.pi * bit_freqs_all / ADC_SAMPLING_RATE
        # Map negative frequencies to the [π, 2π] range
        bit_normalized_freqs_all = np.where(bit_normalized_freqs_all < 0, bit_normalized_freqs_all + 2 * np.pi, bit_normalized_freqs_all)
        bit_normalized_freqs_in_pi = bit_normalized_freqs_all / np.pi
        
        # Sort by normalized frequency for proper display
        bit_sort_idx_norm = np.argsort(bit_normalized_freqs_in_pi)
        axes4[bit].stem(bit_normalized_freqs_in_pi[bit_sort_idx_norm], bit_mag_all[bit_sort_idx_norm], basefmt=' ')
        axes4[bit].set_xlabel("Normalized Frequency (×π rad/sample)")
        axes4[bit].set_ylabel("Magnitude")
        axes4[bit].set_title(f"DFT Spectrum for Bit {bit} (Normalized Frequency)")
        axes4[bit].grid(True, alpha=0.3, axis='y')
        axes4[bit].set_xlim(normalized_xlim) # Align x-axis with the overall DFT plot for comparison.

    fig4.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust layout to make space for the suptitle.
    fig4.savefig('figure4_bit_dfts_normalized.png') # Save the fourth figure.

    plt.show() # Display all figures.


if __name__ == "__main__":
    # --- Simulation Parameters ---
    # This is the main execution block of the script.
    # Here, you can define the parameters for the signal generation and analysis.

    # Define the frequencies (in Hz) and their relative weights (amplitudes)
    # for the input analog signal.
    # For example, {2: 0.5, 5: 0.5} creates a signal composed of a 2 Hz sine
    # wave and a 5 Hz sine wave, each with 50% of the total amplitude.
    frequencies = {2: 0.5, 5: 0.5}

    # Generate the ADC signal based on the specified parameters.
    samples, times = generate_adc_signal(
        frequencies_dict=frequencies,
        duration_periods=10,  # Generate 10 periods of the lowest frequency (2 Hz).
        noise_level=0         # No noise is added to the signal.
    )

    # --- Output Information ---
    # Print summary information about the generated signal.
    print(f"Generated {len(samples)} samples")
    print(f"Sample rate: {ADC_SAMPLING_RATE} Hz")
    print(f"Signal frequencies: {frequencies}")
    print(f"ADC value range: 0 to {ADC_MAX_VALUE}")
    print(f"Generated sample value range: min={samples.min()}, max={samples.max()}")

    # Compute the DFT of the generated ADC signal.
    # A windowing function is not applied here (apply_window=False) to maintain
    # peak clarity for this specific simulation, as the signal is periodic
    # within the sampling window.
    freq_bins, magnitude, phase = compute_dft(samples, apply_window=False)

    # Generate and display all analysis plots.
    plot_all_analysis(samples, times, freq_bins, magnitude, frequencies)
