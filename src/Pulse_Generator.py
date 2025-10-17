import numpy as np

class Pulse_Generator:
    """
    Simulates a realistic pulse generator for a Nyquist Folding Receiver (NYFR).

    Models timing jitter, amplitude distortion, finite rise/fall time,
    and pulse-to-pulse variations.
    """

    def __init__(self,
                 pre_start_val=None,
                 pulse_params=None,
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
        self.pulse_width = pulse_width
        self.amplitude = amplitude
        self.jitter_std = jitter_std
        self.amp_noise_std = amp_noise_std
        self.rise_time = rise_time
        self.fall_time = fall_time
        self.droop_coeff = droop_coeff
        self.baseline_offset = baseline_offset

        if seed is not None:
            np.random.seed(seed)
            
    def set_pulse_params(self, pulse_params=None):
        if pulse_params is None:
            pulse_params = {
                "pulse_width": float,
                "amplitude": 1.0,
                "jitter_std": 0.0,
                "amp_noise_std": 0.0,
                "rise_time": 0.0,
                "fall_time": 0.0,
                "droop_coeff": 0.0,
                "baseline_offset": 0.0,
                "seed": int | None = None               
            }
        

    def generate(self, signal, pre_start_val=None) -> np.ndarray:
        """
        Generate a realistic pulse train

        Returns
        -------
        pulses : np.ndarray
            Generated pulse train with realistic imperfections.
        """
        ideal_pulse_times = self._rising_zero_crossings(signal, pre_start_val)

        # --- Apply timing jitter ---
        if self.jitter_std > 0:
            jitter = np.random.normal(0, self.jitter_std, num_pulses)
            pulse_times = ideal_pulse_times + jitter
        else:
            pulse_times = ideal_pulse_times

        pulses = np.zeros_like(t)

        # --- Generate each pulse individually ---
        for i, t0 in enumerate(pulse_times):
            # Skip pulses outside the time window
            if t0 > t[-1] + self.pulse_width:
                continue

            # Amplitude noise and droop
            amp = self.amplitude * (1 + np.random.normal(0, self.amp_noise_std))
            amp *= np.exp(-self.droop_coeff * t0)

            # Generate trapezoidal pulse (finite rise/fall)
            pulse = np.zeros_like(t)
            start = t0
            end = t0 + self.pulse_width

            rise_mask = (t >= start) & (t < start + self.rise_time)
            flat_mask = (t >= start + self.rise_time) & (t < end - self.fall_time)
            fall_mask = (t >= end - self.fall_time) & (t < end)

            # Linear rise/fall edges
            if self.rise_time > 0:
                pulse[rise_mask] = amp * (t[rise_mask] - start) / self.rise_time
            if self.fall_time > 0:
                pulse[fall_mask] = amp * (1 - (t[fall_mask] - (end - self.fall_time)) / self.fall_time)
            pulse[flat_mask] = amp

            pulses += pulse

        # --- Add baseline offset ---
        pulses += self.baseline_offset

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
            if pre_start_val * signal[0] < 0 and pre_start_val < signal[0]:
                pulses[0] = 1

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0] + 1

        # Keep only rising edges
        for i in zero_crossings:
            if signal[i] > signal[i-1]:
                pulses[i] = 1

        return pulses
