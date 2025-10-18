import numpy as np
from utility import load_settings

class Mixer:
    def __init__(self,
                 rf_signal=None,
                 lo_signal=None,
                 mixer_params=None,
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
            if ( mixer_params is None):
                mixer_config_name = "Default_Mixer_Config"
            self.set_mixer_config_name(mixer_config_name)
        
        self.rf_signal = rf_signal
        self.lo_signal = lo_signal
        self.mixed_signal = None

        if rf_signal is not None and lo_signal is not None:
            self.mixed_signal = self.mix(rf_signal, lo_signal)
            
    # -------------------------------
    # Setters
    # -------------------------------
       
    def set_config_from_file(self, config_file_path=None):
        print("Loading Mixer configuration from file: ", config_file_path)
        mixer_config = load_settings(config_file_path)
        mixer_params = mixer_config.get('mixer_params', None)
        mixer_config_name = mixer_config.get('system_config_name', None)

        self.set_mixer_params(mixer_params)
        self.set_mixer_config_name(mixer_config_name)
        
    def set_mixer_config_name(self, mixer_config_name=None):
        if mixer_config_name is None:
            mixer_config_name = "Mixer_Config_1"
        self.mixer_config_name = mixer_config_name        
        
    def set_mixer_params(self, mixer_params=None):
        if mixer_params is None:
            mixer_params = {
                "conversion_gain": 1.0,
                "lo_leakage": 0.0,
                "rf_leakage": 0.0,
                "nonlinearity_coeff": 0.0,
                "noise_std": 0.0
            }           
        self.mixer_params = mixer_params
        
    # -------------------------------
    # Core functional methods
    # -------------------------------
    
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

    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_mixer_config_name(self):
        return self.mixer_config_name
    
    def get_mixed_signal(self):
        return self.mixed_signal
    
    def get_mixer_params(self):
        return self.mixer_params
    
    def get_rf_signal(self):
        return self.rf_signal
    
    def get_lo_signal(self):
        return self.lo_signal