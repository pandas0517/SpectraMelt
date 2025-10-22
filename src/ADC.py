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
            if ( adc_params is None ):
                adc_config_name = "Default_ADC_Config"
            self.set_adc_config_name(adc_config_name)

        self.conditioned_signal = None
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
        
        self.set_adc_params(adc_params)
        if ( adc_params is None ):
            adc_config_name = "Default_ADC_Config"
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
                "seed": None
            }
        self.rng = np.random.default_rng(adc_params.get('seed', None))          
        self.adc_params = adc_params

    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def _condition_adc_input(self, filtered_signal, v_range=None):
        """
        Simulated Front-End Conditioning Stage

        This function models the analog front-end (AFE) that prepares a signal 
        for ADC sampling. It recenters the waveform around the ADC's midpoint 
        voltage (v_mid) and rescales the amplitude to span the available input 
        range (v_min → v_max) without clipping.

        Args:
            filtered_signal (np.ndarray): 
                The low-pass filtered input signal (typically centered around 0 V).
            v_min (float, optional): 
                Minimum ADC input voltage (lower rail). Defaults to 0.0.
            v_max (float, optional): 
                Maximum ADC input voltage (upper rail). Defaults to 5.0.

        Returns:
            np.ndarray: 
                The conditioned signal, centered at v_mid and scaled to nearly 
                fill the ADC input range.

        Notes:
            - This simulates analog gain and DC bias circuits that shift and 
            scale the waveform before digitization.
            - Amplitude normalization ensures maximum dynamic range without 
            exceeding ADC limits.
        """
        if v_range is None:
            v_range = tuple(self.adc_params.get('v_ref_range', (0, 1)))
        v_min = v_range[0]
        v_max = v_range[1]
        # Compute the ADC mid-point voltage (bias reference)
        v_mid = (v_max + v_min) / 2.0

        # Remove any DC offset so signal is centered around 0 V
        centered = filtered_signal - np.mean(filtered_signal)

        # Find the largest absolute amplitude (peak)
        max_amp = np.max(np.abs(centered))

        if max_amp > 0:
            # Scale signal so its maximum amplitude fits half the ADC range
            scaled = centered * ((v_max - v_min) / 2.0) / max_amp
        else:
            # If signal is constant (flat line), just copy it
            scaled = centered.copy()

        # Re-center the waveform around the ADC midpoint voltage
        return scaled + v_mid


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

        quantizer_signals['mid_times'] = mid_times
        quantizer_signals['quantized_values'] = quantized_values
        quantizer_signals['adc_indices'] = adc_indices
        
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
        conditioned_signal = self._condition_adc_input(signal)
        sh_signals = self._sample_and_hold(conditioned_signal, real_time)
        store_internal_sigs = self.adc_params.get('store_internal_sigs', True)
        if store_internal_sigs:
            self.conditioned_signal = conditioned_signal
            self.sh_signals = sh_signals
        return self._quantizer(sh_signals, real_time) 
        
 
    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_adc_params(self):
        return self.adc_params
    
    def get_conditioned_signal(self):
        return self.conditioned_signal
     
    def get_sh_signals(self):
        return self.sh_signals
    
    def get_sh_output_signal(self):
        return self.sh_signals.get('output_signal', None)
    
    def get_sh_indicies(self):
        return self.sh_signals.get('sh_indicies', None)
    
    def get_sh_sampled_values(self):
        return self.sh_signals.get('sh_sampled_values', None)
    
    def get_quantizer_signals(self):
        return self.quantizer_signals
    
    def get_quantizer_output(self):
        return self.quantizer_signals.get('quantized_values', None)
    
    def get_quantizer_midtimes(self):
        return self.quantizer_signals.get('mid_times', None)
    
    def get_adc_indices(self):
        return self.quantizer_signals.get('adc_indices', None)
    
    def get_adc_config_name(self):
        return self.adc_config_name