import numpy as np
from utils import load_config_from_json, get_logger

class Mixer:
    def __init__(self,
                 rf_signal=None,
                 lo_signal=None,
                 mixer_params=None,
                 log_params=None,
                 mixer_config_name=None,
                 config_file_path=None) -> None:
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
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_mixer_params(mixer_params)
            if mixer_params is None:
                mixer_config_name = "Default_Mixer_Config"
            self.set_mixer_config_name(mixer_config_name)
            self.set_log_params(log_params)
        
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "DEBUG")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
            if config_file_path is not None:
                self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")
        
        self.mixed_signal = None
        if rf_signal is not None and lo_signal is not None:
            self.mixed_signal = self.mix(rf_signal, lo_signal)
            
    # -------------------------------
    # Setters
    # -------------------------------
       
    def set_config_from_file(self, config_file_path):
        mixer_config = load_config_from_json(config_file_path)
        mixer_params = mixer_config.get('mixer_params', None)
        mixer_config_name = mixer_config.get('config_name', "Mixer_Config_1")
        log_params = mixer_config.get('log_params', None)
        
        if mixer_params is None:
            mixer_config_name = "Default_Mixer_Config"

        self.set_log_params(log_params)
        self.set_mixer_params(mixer_params)
        self.set_mixer_config_name(mixer_config_name)
        
    def set_mixer_config_name(self, mixer_config_name):
        self.mixer_config_name = mixer_config_name
        
    def set_log_params(self, log_params=None):
        if log_params is None:
            log_params = {
                "enabled": True,
                "log_file": None,
                "level": "DEBUG",
                "console": True
            }
        self.log_params = log_params        
        
    def set_mixer_params(self, mixer_params=None):
        if mixer_params is None:
            mixer_params = {
                "conversion_gain": 1.0,
                "lo_leakage": 0.0,
                "rf_leakage": 0.0,
                "nonlinearity_coeff": 0.0,
                "noise_std": 0.0,
                "seed": None
            }
        self.rng = np.random.default_rng(mixer_params.get('seed', None))         
        self.mixer_params = mixer_params
        
    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def mix(self, rf_signal: np.ndarray, lo_signal: np.ndarray) -> np.ndarray:
        """Mix RF input with LO, including imperfections."""
        # --- Mixer Nonidealities ---
        conversion_gain = self.mixer_params.get('conversion_gain', 1.0)
        lo_leakage = self.mixer_params.get('lo_leakage', 0.0)
        rf_leakage = self.mixer_params.get('rf_leakage', 0.0)
        nonlinearity_coeff = self.mixer_params.get('nonlinearity_coeff', 0.0)
        noise_std = self.mixer_params.get('noise_std', 0.0)
        
        # Ideal mixing
        mixed = conversion_gain * rf_signal * lo_signal

        # Add imperfections
        mixed += lo_leakage * lo_signal
        mixed += rf_leakage * rf_signal
        mixed += nonlinearity_coeff * (rf_signal ** 3)

        # Add random noise
        if noise_std > 0:
            mixed += self.rng.normal(0, noise_std, size=mixed.shape)

        return mixed

    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_mixer_config_name(self):
        return self.mixer_config_name
    
    def get_mixed_signal(self):
        return self.mixed_signal
    
    def get_mixer_params(self):
        return self.mixer_params
    
    def get_log_params(self):
        return self.log_params