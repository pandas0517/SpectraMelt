import numpy as np
from .utils import load_config_from_json, get_logger

class PulseGenerator:
    """
    Simulates a realistic pulse generator for a Nyquist Folding Receiver (NYFR).

    Models timing jitter, amplitude distortion, finite rise/fall time,
    and pulse-to-pulse variations.
    """

    def __init__(self,
                 signal=None,
                 real_time=None,
                 pre_start_val=None,
                 pulse_params=None,
                 log_params=None,
                 pulse_config_name=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------
        pulse_width : float
            Width of each pulse (seconds).
        amplitude : float
            Peak amplitude of each pulse.
        jitter_std : float
            Std deviation of timing jitter (seconds).
        amp_noise_std : float
            Std deviation of amplitude noise (fractional).
        rise_time : float
            Rise time of the pulse (seconds).
        fall_time : float
            Fall time of the pulse (seconds).
        droop_coeff : float
            Exponential droop factor over time (fraction per second).
        baseline_offset : float
            DC offset of the baseline (volts).
        seed : int, optional
            Random seed for reproducibility.
        """
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_pulse_params(pulse_params)
            if pulse_params is None:
                pulse_config_name = "Default_Pulse_Config"
            self.set_pulse_config_name(pulse_config_name)
            self.set_log_params(log_params)
        
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
            if config_file_path is not None:
                self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")
        
        self.pulse_signal = None
        self.pre_start_val = pre_start_val
        if signal is not None and real_time is not None:
            self.pulse_signal = self.generate(signal, real_time, pre_start_val)

    # -------------------------------
    # Setters
    # -------------------------------

    def set_config_from_file(self, config_file_path):
        pulse_config = load_config_from_json(config_file_path)
        log_params = pulse_config.get('log_params', None)
        pulse_params = pulse_config.get('pulse_params', None)
        pulse_config_name = pulse_config.get('config_name', "Pulse_Config_1")
        
        if pulse_params is None:
            pulse_config_name = "Default_Pulse_Config"

        self.set_log_params(log_params)
        self.set_pulse_params(pulse_params)
        self.set_pulse_config_name(pulse_config_name)

    def set_pulse_config_name(self, pulse_config_name):
        self.pulse_config_name = pulse_config_name
        
    def set_log_params(self, log_params=None):
        if log_params is None:
            log_params = {
                "enabled": True,
                "log_file": None,
                "level": "INFO",
                "console": True
            }
        self.log_params = log_params

    def set_pulse_params(self, pulse_params=None):
        if pulse_params is None:
            pulse_params = {
                "pulse_width": None,
                "amplitude": 1.0,
                "jitter_std": 0.0,
                "amp_noise_std": 0.0,
                "rise_time": 0.0,
                "fall_time": 0.0,
                "droop_coeff": 0.0,
                "baseline_offset": 0.0,
                "seed": None               
            }
        self.rng = np.random.default_rng(pulse_params.get('seed', None))
        self.pulse_params = pulse_params    
        
    # -------------------------------
    # Core functional methods
    # -------------------------------
      
    def generate(self, signal, real_time, pre_start_val=None) -> np.ndarray:
        """
        Generate a realistic pulse train

        Returns
        -------
        pulses : np.ndarray
            Generated pulse train with realistic imperfections.
        """
        # --- Set Pulse Generator Parameters ---
        pulse_width = self.pulse_params.get('pulse_width', None)
        amplitude = self.pulse_params.get('amplitude', 1.0)
        rise_time = self.pulse_params.get('rise_time', 0.0)
        fall_time = self.pulse_params.get('fall_time', 0.0)
        baseline_offset = self.pulse_params.get('baseline_offset', 0.0)
        
        # --- Set Pulse Generator Nonidealities
        jitter_std = self.pulse_params.get('jitter_std', 0.0)
        amp_noise_std = self.pulse_params.get('amp_noise_std', 0.0)
        droop_coeff = self.pulse_params.get('droop_coeff', 0.0)

        if pulse_width is None:
            pulse_width = real_time[1] - real_time[0]
            
        ideal_pulses = self._rising_zero_crossings(signal, pre_start_val)
        ideal_pulse_times = real_time[ideal_pulses != 0]
        num_pulses = len(ideal_pulse_times)

        # --- Apply timing jitter ---
        if jitter_std > 0:
            jitter = self.rng.normal(0, jitter_std, num_pulses)
            pulse_times = ideal_pulse_times + jitter
        else:
            pulse_times = ideal_pulse_times

        pulses = np.zeros_like(real_time)

        # --- Generate each pulse individually ---
        for t0 in pulse_times:
            # Skip pulses outside the time window
            if t0 > real_time[-1] + pulse_width:
                continue

            # Amplitude noise and droop
            amp = amplitude * (1 + self.rng.normal(0, amp_noise_std))
            amp *= np.exp(-droop_coeff * t0)

            # Generate trapezoidal pulse (finite rise/fall)
            pulse = np.zeros_like(real_time)
            start = t0
            end = t0 + pulse_width

            rise_mask = (real_time >= start) & (real_time < start + rise_time)
            flat_mask = (real_time >= start + rise_time) & (real_time < end - fall_time)
            fall_mask = (real_time >= end - fall_time) & (real_time < end)

            # Linear rise/fall edges
            if rise_time > 0:
                pulse[rise_mask] = amp * (real_time[rise_mask] - start) / rise_time
            if fall_time > 0:
                pulse[fall_mask] = amp * (1 - (real_time[fall_mask] - (end - fall_time)) / fall_time)
            pulse[flat_mask] = amp

            pulses += pulse

        # --- Add baseline offset ---
        pulses += baseline_offset

        return pulses

    def _rising_zero_crossings(self, signal, pre_start_val=None):
        """
        Generate a pulse train from the rising zero crossings of a signal.

        Parameters
        ----------
        signal : np.ndarray
            Analog waveform (e.g., LO signal).
        start_val : float, optional
            Optional value to check for edge crossing at the start of t.

        Returns
        -------
        pulses : np.ndarray
            Array of 0s and 1s, with 1s at rising zero crossings.
        """
        pulses = np.zeros_like(signal)

        # Handle edge at the start of the time vector
        if pre_start_val is not None:
            if np.signbit(pre_start_val) != np.signbit(signal[0]) and pre_start_val < signal[0]:
                pulses[0] = 1

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0] + 1

        # Keep only rising edges
        for i in zero_crossings:
            if signal[i] > signal[i-1]:
                pulses[i] = 1

        return pulses
    
    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_pulse_config_name(self):
        return self.pulse_config_name
    
    def get_pulse_signal(self):
        return self.pulse_signal

    def get_pulse_params(self):
        return self.pulse_params
    
    def get_log_params(self):
        return self.log_params
    
    def get_pre_start_val(self):
        return self.pre_start_val