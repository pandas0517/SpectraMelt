import numpy as np

class Mixer:
    def __init__(self,
                 conversion_gain: float = 0.8,
                 lo_leakage: float = 0.02,
                 rf_leakage: float = 0.01,
                 nonlinearity_coeff: float = 0.001,
                 noise_std: float = 0.001):
        """
        Simulates a realistic RF/baseband mixer.

        Parameters
        ----------
        conversion_gain : float
            Mixer conversion gain (typ. < 1).
        lo_leakage : float
            Fraction of LO leaking to output.
        rf_leakage : float
            Fraction of RF leaking to output.
        nonlinearity_coeff : float
            Coefficient for cubic nonlinearity.
        noise_std : float
            Std. dev. of Gaussian noise added to output.
        """
        self.conversion_gain = conversion_gain
        self.lo_leakage = lo_leakage
        self.rf_leakage = rf_leakage
        self.nonlinearity_coeff = nonlinearity_coeff
        self.noise_std = noise_std

    def mix(self, rf_signal: np.ndarray, lo_signal: np.ndarray) -> np.ndarray:
        """Mix RF input with LO, including imperfections."""
        # Ideal mixing
        mixed = self.conversion_gain * rf_signal * lo_signal

        # Add imperfections
        mixed += self.lo_leakage * lo_signal
        mixed += self.rf_leakage * rf_signal
        mixed += self.nonlinearity_coeff * (rf_signal ** 3)

        # Add random noise
        if self.noise_std > 0:
            mixed += np.random.normal(0, self.noise_std, size=mixed.shape)

        return mixed
