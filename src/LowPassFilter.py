import numpy as np
from scipy import signal
from utility import load_settings

class LowPassFilter:
    """
    Simulates a realistic low-pass filter (LPF) stage for baseband or IF signals.
    
    Supports multiple filter types (Butterworth, Chebyshev, Bessel, Elliptic),
    configurable order, cutoff frequency, and optional additive noise.
    """

    def __init__(self,
                 signal_in=None,
                 real_time=None,
                 lpf_params=None,
                 lpf_config_name=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------
        signal_in : np.ndarray, optional
            Input signal to be filtered.
        lpf_params : dict, optional
            Dictionary of filter parameters.
        lpf_config_name : str, optional
            Name of the filter configuration.
        config_file_path : Path or str, optional
            Path to configuration file containing filter settings.
        """
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_lpf_params(lpf_params)
            if lpf_config_name is None:
                lpf_config_name = "Default_LPF_Config"
            self.set_lpf_config_name(lpf_config_name)

        self.signal_out = None
        if signal_in is not None and real_time is not None:
            self.signal_out = self.apply_filter(signal_in, real_time)

    # -------------------------------
    # Setters
    # -------------------------------

    def set_config_from_file(self, config_file_path=None):
        print("Loading LPF configuration from file:", config_file_path)
        lpf_config = load_settings(config_file_path)
        lpf_params = lpf_config.get('lpf_params', None)
        lpf_config_name = lpf_config.get('config_name', None)

        self.set_lpf_params(lpf_params)
        self.set_lpf_config_name(lpf_config_name)

    def set_lpf_config_name(self, lpf_config_name=None):
        if lpf_config_name is None:
            lpf_config_name = "LPF_Config_1"
        self.lpf_config_name = lpf_config_name

    def set_lpf_params(self, lpf_params=None):
        if lpf_params is None:
            lpf_params = {
                "filter_type": "butter",      # 'butter', 'cheby1', 'cheby2', 'bessel', 'ellip'
                "order": 4,
                "cutoff_freq": 100,         # Hz
                "ripple_db": 1.0,             # Used for cheby/ellip
                "atten_db": 40.0,             # Stopband attenuation (cheby2/ellip)
                "noise_std": 0.0,
                "seed": None
            }
        self.rng = np.random.default_rng(lpf_params.get('seed', None))
        self.lpf_params = lpf_params

    # -------------------------------
    # Core Functional Methods
    # -------------------------------

    def apply_filter(self, signal_in: np.ndarray, real_time: np.ndarray) -> np.ndarray:
        """Apply the configured low-pass filter to the input signal."""
        # --- Get Real Time Sample Frequency ---
        dt = np.mean(np.diff(real_time))  # average time step
        sim_freq = 1.0 / dt
        # --- Set Filter Parameters ---
        fc = self.lpf_params.get('cutoff_freq', 100)
        filter_type = self.lpf_params.get('filter_type', "butter")
        order = self.lpf_params.get('order', 4)
        
        # --- Set Filter Nonidealities ---
        ripple_db = self.lpf_params.get('ripple_db', 1.0)
        atten_db = self.lpf_params.get('atten_db', 40.0)
        noise_std = self.lpf_params.get('noise_std', 0.0)

        # Normalize cutoff (Nyquist = fs/2)
        wn = fc / (sim_freq / 2)

        # Select filter design
        if filter_type == 'butter':
            b, a = signal.butter(order, wn, btype='low', analog=False)
        elif filter_type == 'cheby1':
            b, a = signal.cheby1(order, ripple_db, wn, btype='low')
        elif filter_type == 'cheby2':
            b, a = signal.cheby2(order, atten_db, wn, btype='low')
        elif filter_type == 'ellip':
            b, a = signal.ellip(order, ripple_db, atten_db, wn, btype='low')
        elif filter_type == 'bessel':
            b, a = signal.bessel(order, wn, btype='low', norm='phase')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # Apply zero-phase filtering to avoid group delay
        filtered = signal.filtfilt(b, a, signal_in)

        # Add optional Gaussian noise
        if noise_std > 0:
            filtered += self.rng.normal(0, noise_std, size=filtered.shape)

        return filtered

    # -------------------------------
    # Getters
    # -------------------------------

    def get_lpf_params(self):
        return self.lpf_params

    def get_signal_out(self):
        return self.signal_out

    def get_lpf_config_name(self):
        return self.lpf_config_name
