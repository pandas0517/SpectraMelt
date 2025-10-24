import numpy as np
from utility import load_settings

class ADC:
    def __init__(self,
                 input_signal=None,
                 real_time=None,
                 adc_params=None,
                 adc_config_name=None,
                 config_file_path=None) -> None:
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_adc_params(adc_params)
            if adc_params is None:
                adc_config_name = "Default_ADC_Config"
            self.set_adc_config_name(adc_config_name)

        self.conditioned_signals = None
        self.sh_signals = None
        self.quantizer_signals = None
        
        if input_signal is not None and real_time is not None:
            self.quantizer_signals = self.analog_to_digital(input_signal, real_time)
            
    # -------------------------------
    # Setters
    # -------------------------------
    
    def set_config_from_file(self, config_file_path):
        print("Loading adc configuration from file: ", config_file_path)
        adc_config = load_settings(config_file_path)
        adc_params = adc_config.get('adc_params', None)
        adc_config_name = adc_config.get('config_name', None)  

        if adc_params is None:
            adc_config_name = "Default_ADC_Config"
            
        self.set_adc_params(adc_params)
        self.set_adc_config_name(adc_config_name)
        
    def set_adc_config_name(self, adc_config_name=None):
        if adc_config_name is None:
            adc_config_name = "ADC_Config_1"
        self.adc_config_name = adc_config_name

    def set_adc_params(self, adc_params=None):
        if adc_params is None:
            adc_params = {
                "store_internal_sigs": True,
                "adc_samp_freq": 100,
                "allow_clipping": True,
                "v_ref_range": (0, 1),
                "num_bits": 8,
                "thermal_noise_std_dev": 0.0,
                "non_linearity_mode": None,
                "alpha": 0.0,
                "threshold": 1.0,
                "jitter_std": 0.0,
                "acquisition_time_constant": 0.0,
                "hold_noise_std": 0.0,
                "transient_mode": "none",
                "truncate_transients": False,
                "transient_fraction": 0.1,
                "detection_window": 0.05,
                "stability_threshold": 0.01,
                "seed": None
            }
        self.rng = np.random.default_rng(adc_params.get('seed', None))          
        self.adc_params = adc_params

    # -------------------------------
    # Core functional methods
    # -------------------------------

    def _condition_adc_input(self, filtered_signal: np.ndarray, real_time: np.ndarray) -> np.ndarray:
        """
        Simulated Front-End Conditioning Stage

        Centers and scales the filtered signal for ADC input, with optional
        transient suppression (auto or fixed).

        ADC config parameters:
        ----------------------
        v_ref_range : (float, float)  → (v_min, v_max)
        transient_mode : "auto" | "fixed" | "none"
        transient_fraction : float    → used if mode="fixed"
        detection_window : float      → used if mode="auto"
        stability_threshold : float   → used if mode="auto"
        truncate_transients : bool

        Returns
        -------
        np.ndarray:
            Fully conditioned signal matching input length.
        """

        if not isinstance(filtered_signal, np.ndarray):
            raise TypeError("filtered_signal must be a NumPy array")

        n = filtered_signal.size
        if n == 0:
            return filtered_signal

        # === ADC Voltage Range & Settings ===
        v_min, v_max = self.adc_params.get('v_ref_range', (0.0, 1.0))
        transient_mode = self.adc_params.get('transient_mode', 'none').lower()
        transient_fraction = self.adc_params.get('transient_fraction', 0.05)
        detection_window = self.adc_params.get('detection_window', 0.05)
        stability_threshold = self.adc_params.get('stability_threshold', 0.01)
        truncate_transients = self.adc_params.get('truncate_transients', False)

        # Make a safe working copy
        signal = filtered_signal.copy()

        # =================================================================
        # === Determine Transient Region Based on Selected Mode ===
        # =================================================================
        if transient_mode == "none":
            skip_start = 0
            skip_end = 0

        elif transient_mode == "fixed":
            skip = int(n * transient_fraction)
            skip_start = skip_end = min(skip, n // 2)

        elif transient_mode == "auto":
            window_len = max(1, int(n * detection_window))
            num_windows = n // window_len

            variances = np.array([
                np.var(signal[i*window_len:(i+1)*window_len])
                for i in range(num_windows)
            ])

            mean_var = np.mean(variances[-3:])  # assume last windows are steady
            relative_change = np.abs(np.diff(variances) / (mean_var + 1e-12))

            start_idx = np.argmax(relative_change < stability_threshold)
            end_idx = num_windows - np.argmax(relative_change[::-1] < stability_threshold) - 1

            skip_start = int(start_idx * window_len)
            skip_end = int(n - end_idx * window_len)

            skip_start = min(skip_start, n // 2)
            skip_end = min(skip_end, n // 2)

        else:
            raise ValueError("transient_mode must be 'auto', 'fixed', or 'none'")

        # =================================================================
        # === Conditioning Based Only on Steady-State Region ===
        # =================================================================
        steady = signal[skip_start:n - skip_end] if skip_start < n - skip_end else signal
        if steady.size == 0:
            steady = signal.copy()

        v_mid = (v_max + v_min) / 2.0
        centered = steady - np.mean(steady)
        max_amp = np.max(np.abs(centered))

        if max_amp > 0:
            scale_factor = ((v_max - v_min) / 2.0) / max_amp
            scaled_centered = centered * scale_factor
        else:
            scaled_centered = centered.copy()

        # =================================================================
        # === Reconstruct Full Output Signal ===
        # =================================================================
        if truncate_transients:
            conditioned_signal = scaled_centered + v_mid
            corresponding_time = real_time[skip_start:n - skip_end]
        else:
            conditioned_signal = np.full_like(signal, v_mid)
            conditioned_signal[skip_start:n - skip_end] = scaled_centered + v_mid
            corresponding_time = real_time
            
        # --- Compute corresponding frequency axis (for FFT-based analysis) ---
        num_samples = len(corresponding_time)
        total_time = corresponding_time[-1] - corresponding_time[0]
        corresponding_freq = np.linspace(
            -0.5 * num_samples / total_time,
            0.5 * num_samples / total_time,
            num_samples,
            endpoint=False
        )

        conditioned = {
            "signal": conditioned_signal,
            "time": corresponding_time,
            "freq": corresponding_freq,
            "total_time": total_time
        }
        return conditioned

    def _quantizer(self, sh_signals, real_time):
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
        quantizer_signals = {}
        
        # --- Get Real Time Sample Frequency ---
        dt = np.mean(np.diff(real_time))  # average time step
        sim_freq = 1.0 / dt
        
        start_time = real_time[0]
        
        # --- Internal Sample and Hold Signals ---
        output_signal = sh_signals.get('output_signal')
        indices = sh_signals.get('indices')
        
        # --- Quantizer Parameters ---
        v_ref_range = self.adc_params.get('v_ref_range', (0, 1))
        num_levels = 2**self.adc_params.get('num_bits', 8)
        
        # --- Quantizer Nonidealities ---
        thermal_noise_std_dev = self.adc_params.get('thermal_noise_std_dev', 0)
        
        quantization_step = (v_ref_range[1] - v_ref_range[0]) / num_levels

        quantized_values = np.zeros(len(indices))
        mid_times = np.zeros(len(indices))
        adc_indices = np.zeros(len(indices), dtype=int)

        for i in range(len(indices)):
            start_index = indices[i]
            end_index = (indices[i+1]
                         if i <len(indices)-1
                         else len(output_signal))
            mid_index = start_index + (end_index - start_index)//2

            mid_times[i] = mid_index / sim_freq
            sample_value = output_signal[mid_index]

            # Add thermal noise if specified
            if thermal_noise_std_dev > 0:
                sample_value += self.rng.normal(0, thermal_noise_std_dev)

            # Clip to ADC range
            sample_value = np.clip(sample_value, v_ref_range[0], v_ref_range[1])

            # Quantize to integer ADC code
            index = int(np.floor((sample_value - v_ref_range[0]) / quantization_step))
            index = np.clip(index, 0, num_levels - 1)
            adc_indices[i] = index

            # Convert back to voltage and round to ADC precision
            quantized_values[i] = v_ref_range[0] + (index + 0.5) * quantization_step
            # Round voltage to nearest step for exact n-bit precision
            quantized_values[i] = (np.round((quantized_values[i] - v_ref_range[0]) / quantization_step) 
                                   * quantization_step + v_ref_range[0])
        if start_time != 0:
            mid_times += start_time

        # --- Compute sampled frequency axis (for FFT-based analysis) ---
        num_samples = len(mid_times)
        total_time = mid_times[-1] - mid_times[0]
        if total_time <= 0:
            total_time = 1.0 / sim_freq  # fallback if single sample
        sampled_freq = np.linspace(
            -0.5 * num_samples / total_time,
             0.5 * num_samples / total_time,
             num_samples,
             endpoint=False
        )

        quantizer_signals['mid_times'] = mid_times
        quantizer_signals['quantized_values'] = quantized_values
        quantizer_signals['adc_indices'] = adc_indices
        quantizer_signals['sampled_frequency'] = sampled_freq
        
        return quantizer_signals

    def _sample_and_hold(self, signal, real_time):
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
        sh_signals = {}
        v_min = np.min(signal)
        v_max = np.max(signal)
        adc_samp_freq = self.adc_params.get('adc_samp_freq', 100)
        
        # --- Get Real Time Sample Frequency ---
        dt = np.mean(np.diff(real_time))  # average time step
        sim_freq = 1.0 / dt
                    
        # --- Sample and Hold Nonidealities ---
        jitter_std = self.adc_params.get('jitter_std', 0.0)
        non_linearity_mode = self.adc_params.get('non_linearity_mode', None)
        alpha = self.adc_params.get('alpha', 0)
        threshold = self.adc_params.get('threshold', 1.0)
        acquisition_time_constant = self.adc_params.get('acquisition_time_constant', 0.0)
        hold_noise_std = self.adc_params.get('hold_noise_std', 0.0)
        
        num_samples_sh = int(np.floor(len(signal) * adc_samp_freq / sim_freq))

        # Jittered sampling indices
        ideal_sh_indices = np.arange(num_samples_sh) * (sim_freq / adc_samp_freq)
        jitter_indices = self.rng.normal(0, jitter_std * sim_freq, num_samples_sh)
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
                        1 - np.exp(-(j - start_index) / (sim_freq * acquisition_time_constant))
                    )
                current_value = sampled_values[i]
            else:
                output_signal[start_index:end_index] = sampled_values[i]

            # Add hold noise once per hold interval
            if hold_noise_std > 0:
                noise = self.rng.normal(0, hold_noise_std)
                output_signal[start_index:end_index] += noise
        sh_signals['output_signal'] = output_signal
        sh_signals['indices'] = sh_indices
        sh_signals['sampled_values'] = sampled_values
        
        return sh_signals

    def analog_to_digital(self, signal, real_time):
        conditioned_signals = self._condition_adc_input(signal, real_time)
        conditioned_signal = conditioned_signals.get('signal')
        conditioned_time = conditioned_signals.get('time')
        sh_signals = self._sample_and_hold(conditioned_signal, conditioned_time)
        store_internal_sigs = self.adc_params.get('store_internal_sigs', True)
        if store_internal_sigs:
            self.conditioned_signals = conditioned_signals
            self.conditioned_time = conditioned_time
            self.sh_signals = sh_signals
        return self._quantizer(sh_signals, conditioned_time) 
        
 
    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_adc_params(self):
        return self.adc_params
    
    def get_conditioned_signals(self):
        return self.conditioned_signals
     
    def get_sh_signals(self):
        return self.sh_signals
    
    def get_sh_output_signal(self):
        return self.sh_signals.get('output_signal') if self.sh_signals else None
    
    def get_sh_indicies(self):
        return self.sh_signals.get('sh_indicies') if self.sh_signals else None
    
    def get_sh_sampled_values(self):
        return self.sh_signals.get('sh_sampled_values') if self.sh_signals else None
    
    def get_quantizer_signals(self):
        return self.quantizer_signals
    
    def get_sampled_frequency(self):
        return self.quantizer_signals.get('sampled_frequency') if self.quantizer_signals else None
    
    def get_quantizer_output(self):
        return self.quantizer_signals.get('quantized_values') if self.quantizer_signals else None
    
    def get_quantizer_midtimes(self):
        return self.quantizer_signals.get('mid_times') if self.quantizer_signals else None
    
    def get_adc_indices(self):
        return self.quantizer_signals.get('adc_indices') if self.quantizer_signals else None
    
    def get_adc_config_name(self):
        return self.adc_config_name