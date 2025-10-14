import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def realistic_adc_quantizer(
    sh_output_signal,
    sh_indices,
    fs_in,
    num_bits,
    v_ref_min,
    v_ref_max,
    thermal_noise_std_dev=0
):
    """
    Simulates a realistic ADC quantizer by sampling the S&H output at
    the midpoint of each hold interval, and returns both quantized values
    and corresponding n-bit codes with correct ADC precision.

    Args:
        sh_output_signal (np.ndarray): Output from the S&H circuit.
        sh_indices (np.ndarray): Indices where each S&H sample starts.
        fs_in (float): Original analog signal sampling frequency (Hz).
        num_bits (int): Number of ADC bits.
        v_ref_min (float): Minimum ADC reference voltage.
        v_ref_max (float): Maximum ADC reference voltage.
        thermal_noise_std_dev (float): Standard deviation of thermal noise.

    Returns:
        tuple:
            - quantized_values (np.ndarray): ADC quantized values at midpoints (volts).
            - mid_times (np.ndarray): Time points corresponding to each quantized value.
            - adc_indices (np.ndarray): Integer n-bit ADC codes for each sample.
    """

    num_levels = 2**num_bits
    quantization_step = (v_ref_max - v_ref_min) / num_levels

    quantized_values = np.zeros(len(sh_indices))
    mid_times = np.zeros(len(sh_indices))
    adc_indices = np.zeros(len(sh_indices), dtype=int)

    for i in range(len(sh_indices)):
        start_index = sh_indices[i]
        end_index = sh_indices[i+1] if i < len(sh_indices)-1 else len(sh_output_signal)
        mid_index = start_index + (end_index - start_index)//2

        mid_times[i] = mid_index / fs_in
        sample_value = sh_output_signal[mid_index]

        # Add thermal noise if specified
        if thermal_noise_std_dev > 0:
            sample_value += np.random.normal(0, thermal_noise_std_dev)

        # Clip to ADC range
        sample_value = np.clip(sample_value, v_ref_min, v_ref_max)

        # Quantize to integer ADC code
        index = int(np.floor((sample_value - v_ref_min) / quantization_step))
        index = np.clip(index, 0, num_levels - 1)
        adc_indices[i] = index

        # Convert back to voltage and round to ADC precision
        quantized_values[i] = v_ref_min + (index + 0.5) * quantization_step
        # Round voltage to nearest step for exact n-bit precision
        quantized_values[i] = np.round((quantized_values[i] - v_ref_min) / quantization_step) * quantization_step + v_ref_min

    return quantized_values, mid_times, adc_indices

def realistic_sample_and_hold(
    signal,
    fs_in,
    fs_sh,
    non_linearity_mode=None,
    alpha=0.05,
    threshold=1.0,
    jitter_std=0.0,
    acquisition_time_constant=0.0,
    hold_noise_std=0.0,
    v_min=None,
    v_max=None
):
    """
    Simulates a realistic sample-and-hold circuit with optional voltage-preserving non-linearity.

    Args:
        signal (np.ndarray): Input analog signal.
        fs_in (float): Input signal sampling frequency.
        fs_sh (float): Sample-and-hold frequency.
        non_linearity_mode (str or None): 'tanh', 'cubic', 'hard_clip', or None.
        alpha (float): Scaling for tanh or cubic distortion.
        threshold (float): Threshold for hard clipping.
        jitter_std (float): Standard deviation of sampling time jitter (seconds).
        acquisition_time_constant (float): Time constant for finite acquisition time (seconds).
        hold_noise_std (float): Noise added once per hold interval.
        v_min (float or None): Minimum voltage for normalization. Defaults to min(signal).
        v_max (float or None): Maximum voltage for normalization. Defaults to max(signal).

    Returns:
        tuple: (output_signal, sh_indices, sampled_values)
    """
    if v_min is None: v_min = np.min(signal)
    if v_max is None: v_max = np.max(signal)

    T_sh = 1.0 / fs_sh
    num_samples_sh = int(np.floor(len(signal) * fs_sh / fs_in))

    # Jittered sampling indices
    ideal_sh_indices = np.arange(num_samples_sh) * (fs_in / fs_sh)
    jitter_indices = np.random.normal(0, jitter_std * fs_in, num_samples_sh)
    sh_indices = np.clip(ideal_sh_indices + jitter_indices, 0, len(signal) - 1).astype(int)

    # Sample values at jittered points
    sampled_values = signal[sh_indices]

    # Apply non-linearity if requested
    if non_linearity_mode is not None:
        if non_linearity_mode == "tanh":
            # Voltage-preserving normalized tanh
            signal_norm = 2 * (sampled_values - v_min) / (v_max - v_min) - 1
            distorted_norm = np.tanh(alpha * signal_norm) / np.tanh(alpha)
            sampled_values = 0.5 * (distorted_norm + 1) * (v_max - v_min) + v_min
        elif non_linearity_mode == "cubic":
            sampled_values = sampled_values - (alpha / 3.0) * (sampled_values**3)
        elif non_linearity_mode == "hard_clip":
            sampled_values = np.clip(sampled_values, v_min + threshold, v_max - threshold)
        else:
            raise ValueError(f"Unknown non-linearity mode: {non_linearity_mode}")

    # Initialize output
    output_signal = np.zeros(len(signal))
    current_value = 0.0

    for i in range(num_samples_sh):
        start_index = sh_indices[i]
        end_index = sh_indices[i + 1] if i < num_samples_sh - 1 else len(signal)

        # Simulate finite acquisition by ramping
        if acquisition_time_constant > 0:
            for j in range(start_index, end_index):
                output_signal[j] = current_value + (sampled_values[i] - current_value) * (
                    1 - np.exp(-(j - start_index) / (fs_in * acquisition_time_constant))
                )
            current_value = sampled_values[i]
        else:
            output_signal[start_index:end_index] = sampled_values[i]

        # Add hold noise once per hold interval
        if hold_noise_std > 0:
            noise = np.random.normal(0, hold_noise_std)
            output_signal[start_index:end_index] += noise

    return output_signal, sh_indices, sampled_values

def sample_signals(self, data=None, update_sampled_time=False, sample_rate=None, points_per_second=None, t=None):
    if points_per_second is None:
        points_per_second = self.real_points_per_second
    if sample_rate is None:
        sample_rate = self.system_params['adc_clock_freq']
    if t is None:
        t = self.real_t
    clock_ticks = int(points_per_second / sample_rate)
    sampled_data_list = []
    sampled_time_list = []
    for i in range(0, t.size, clock_ticks):
        if data is not None:
            sampled_data_list.append(data[i])
        if update_sampled_time:
            sampled_time_list.append(t[i])
    if update_sampled_time:
        self.sampled_t = np.array(sampled_time_list)
        self.sampled_tf = np.linspace(-self.system_params['adc_clock_freq']/2,
                                    self.system_params['adc_clock_freq']/2,
                                    len(sampled_time_list),
                                    endpoint=False)
    sampled_data = np.array(sampled_data_list)

    return sampled_data

# Example Usage:
if __name__ == "__main__":
    # Signal parameters
    fs_in = 44100  # Input sampling frequency (e.g., 44.1 kHz)
    fs_sh = 500    # Sample and hold frequency (e.g., 500 Hz)
    duration = 1.0 # seconds
    t_in = np.linspace(0, duration, int(fs_in * duration), endpoint=False)

    # Generate an input signal (sum of sinusoids)
    f1, f2 = 50, 150
    signal_in = 0.7 * np.sin(2 * np.pi * f1 * t_in) + 0.3 * np.sin(2 * np.pi * f2 * t_in) + 2.5 # Sine wave between 0V to 5V

    # ADC parameters
    bits = 8
    v_ref_min_adc = 0.0
    v_ref_max_adc = 5.0
    thermal_noise_strength = 0.05 # Standard deviation of thermal noise
    
    # Simulate the realistic sample and hold
    output_signal, sh_indices, sampled_values = realistic_sample_and_hold(
        signal_in,
        fs_in,
        fs_sh,
        non_linearity_mode="tanh",
        alpha=0.5,
        jitter_std=10e-6,
        acquisition_time_constant=1e-5,
        hold_noise_std=0.01,
        v_min=v_ref_min_adc,
        v_max=v_ref_max_adc
    )
    
    sampled_time = t_in[sh_indices]

    # --- Plotting the S&H results ---
    plt.figure(figsize=(15, 8))
    plt.plot(t_in, signal_in, label='Original Analog Signal', color='gray', linestyle='--')
    plt.plot(t_in, output_signal, label='Realistic S&H Output', color='red', alpha=0.7)
    plt.plot(t_in[sh_indices], signal_in[sh_indices], 'x', markersize=8, color='blue', label='Jittered Sample Points')
    plt.title('Realistic Sample and Hold with Non-linear Distortion and Imperfections')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.ylim(1.5, 3.5)
    plt.xlim(0.1, 0.15) # Zoom in to see the effects clearly
    plt.show()

    # Quantize the signal with thermal noise
    quantized_signal_with_noise, mid_times, adc_codes_with_noise = realistic_adc_quantizer(
        sh_output_signal=output_signal, 
        sh_indices=sh_indices,
        fs_in=fs_in,
        num_bits=bits,
        v_ref_min=v_ref_min_adc,
        v_ref_max=v_ref_max_adc,
        thermal_noise_std_dev=thermal_noise_strength
    )

    # Quantize the signal without noise for comparison
    quantized_signal_ideal, _, adc_codes_ideal = realistic_adc_quantizer(
        sh_output_signal=output_signal, 
        sh_indices=sh_indices,
        fs_in=fs_in,
        num_bits=bits,
        v_ref_min=v_ref_min_adc,
        v_ref_max=v_ref_max_adc,
        thermal_noise_std_dev=0
    )

    print(f"Original signal sample: {signal_in[100]:.4f}V")
    print("First 10 midpoint times:", mid_times[:10])
    print("First 10 quantized voltages with noise:", quantized_signal_with_noise[:10])
    print("First 10 ADC n-bit codes with noise:", adc_codes_with_noise[:10])
    print("First 10 quantized voltages:", quantized_signal_ideal[:10])
    print("First 10 ADC n-bit codes:", adc_codes_ideal[:10])
    
    # --- Plotting the Quantization results ---
    # --- Side-by-side subplots for comparison ---
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(t_in, output_signal, color='black')
    axs[0].set_title('Original S&H Output Signal')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].grid(True)
    axs[0].set_ylim(1.5, 3.5)
    axs[0].set_xlim(0.1, 0.15) # Zoom in to see the effects clearly
    
    axs[1].step(mid_times, quantized_signal_ideal, color='green', where='mid')
    axs[1].set_title('Quantized Signal (Ideal)')
    axs[1].set_ylabel('Voltage (V)')
    axs[1].grid(True)
    axs[1].set_ylim(1.5, 3.5)
    axs[1].set_xlim(0.1, 0.15) # Zoom in to see the effects clearly
    
    axs[2].step(mid_times, quantized_signal_with_noise, color='orange', where='mid')
    axs[2].set_title('Quantized Signal (With Thermal Noise)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Voltage (V)')
    axs[2].grid(True)
    axs[2].set_ylim(1.5, 3.5)
    axs[2].set_xlim(0.1, 0.15) # Zoom in to see the effects clearly
    
    plt.tight_layout()
    plt.ylim(1.5, 3.5)
    plt.xlim(0.1, 0.15) # Zoom in to see the effects clearly
    plt.show()