import numpy as np

class LocalOscillator:
    def __init__(self,
                 freq_hz: float,
                 amplitude: float = 1.0,
                 phase_offset: float = 0.0,
                 phase_noise_std: float = 1e-3,
                 amp_noise_std: float = 1e-3,
                 freq_drift_ppm: float = 0.0,
                 harmonic_distortion: float = 0.0):
        """
        Simulates a realistic local oscillator.

        Parameters
        ----------
        freq_hz : float
            Nominal LO frequency.
        amplitude : float
            LO amplitude.
        phase_offset : float
            Static phase offset (radians).
        phase_noise_std : float
            Std dev of random phase noise (radians).
        amp_noise_std : float
            Std dev of amplitude noise (fraction of amplitude).
        freq_drift_ppm : float
            Frequency drift in parts per million.
        harmonic_distortion : float
            Fractional 2nd harmonic distortion amplitude.
        """
        self.freq_hz = freq_hz
        self.amplitude = amplitude
        self.phase_offset = phase_offset
        self.phase_noise_std = phase_noise_std
        self.amp_noise_std = amp_noise_std
        self.freq_drift_ppm = freq_drift_ppm
        self.harmonic_distortion = harmonic_distortion

    def generate(self, t: np.ndarray) -> np.ndarray:
        """Generate the LO waveform with imperfections."""
        # Frequency drift (slow variation)
        drift = self.freq_hz * (1 + np.random.normal(0, self.freq_drift_ppm * 1e-6))

        # Phase noise
        phase_noise = np.cumsum(np.random.normal(0, self.phase_noise_std, len(t)))

        # Amplitude noise
        amp_noise = 1 + np.random.normal(0, self.amp_noise_std, len(t))

        # Base LO waveform
        lo = self.amplitude * amp_noise * np.sin(2 * np.pi * drift * t + self.phase_offset + phase_noise)

        # Add harmonic distortion
        if self.harmonic_distortion > 0:
            lo += self.harmonic_distortion * np.sin(4 * np.pi * drift * t + self.phase_offset)

        return lo