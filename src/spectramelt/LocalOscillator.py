import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union
from .utils import load_config_from_json, get_logger


@dataclass(frozen=True)
class LOEffects:
    phase_noise: np.ndarray | None = None
    pre_phase_noise: np.ndarray | None = None
    amp_noise: np.ndarray | None = None
    pre_amp_noise: np.ndarray | None = None
    

class LocalOscillator:
    """
    Represents a realistic local oscillator (LO) with optional modulation, noise,
    frequency drift, and harmonic distortion.
    """

    def __init__(self,
                 lo_params=None,
                 log_params=None,
                 config_name=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------
        lo_params : dict
            {
                'freq': carrier frequency (Hz),
                'amp': amplitude,
                'phase': initial phase (radians),
                'mod_enable': bool,                     # enable phase modulation
                'phase_freq': modulation frequency (Hz),
                'phase_delta': phase deviation (radians),
                'phase_offset': phase offset (radians),

                'phase_noise_std': float,                # std dev of random phase noise (radians)
                'amp_noise_std': float,                  # std dev of amplitude noise (fraction of amplitude)
                'freq_drift_ppm': float,                 # frequency drift in parts per million
                'harmonic_distortion': float             # fractional 2nd harmonic amplitude
            }
        t : np.ndarray
            Time vector for signal generation.
        """
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_lo_params(lo_params)
            if lo_params is None:
                config_name = "Default_LO_Config"
            self.set_config_name(config_name)
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

        self.pre_start_lo = None
        self.phase_mod = None 
 
    # -------------------------------
    # Setters
    # -------------------------------
       
    def set_config_from_file(self, config_file_path):
        lo_config = load_config_from_json(config_file_path)
        lo_params = lo_config.get('lo_params', None)
        config_name = lo_config.get('config_name', "LO_Config_1")
        log_params = lo_config.get('log_params', None)
        
        if lo_params is None:
            config_name = "Default_LO_Config"
        
        self.set_lo_params(lo_params)
        self.set_config_name(config_name)
        self.set_log_params(log_params)
        
    def set_config_name(self, config_name):
        self.config_name = config_name
        
    def set_log_params(self, log_params=None):
        if log_params is None:
            log_params = {
                "enabled": True,
                "log_file": None,
                "level": "INFO",
                "console": True
            }
        self.log_params = log_params         
        
    def set_lo_params(self, lo_params=None):
        if lo_params is None:
            lo_params = {
                "amp": 1,
                "freq": 100,
                "phase": 0,
                "mod_enabled": True,
                "phase_delta": 0.08,
                "phase_freq": 0.1,
                "phase_offset": 0.0,
                "phase_noise_std": 0.0,
                "amp_noise_std": 0.0,
                "freq_drift_ppm": 0.0,
                "harmonic_distortion": 0.0,
                "seed": None
            }
        self.rng = np.random.default_rng(lo_params.get('seed', None))           
        self.lo_params = lo_params
        
    # -------------------------------
    # Core functional methods
    # -------------------------------

    def generate_signal(self, real_time,
                        return_phase_mod=False,
                        return_pre_start=False,
                        return_effects=False
                        ) -> Union[np.ndarray,
                                   Tuple[np.ndarray, np.ndarray],
                                   Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   Tuple[np.ndarray, np.ndarray, np.ndarray, LOEffects]]:
        # --- Local Oscillator Base Signal ---
        f0 = self.lo_params.get('freq', 100)
        A = self.lo_params.get('amp', 1)
        phase = self.lo_params.get('phase', 0)
        pre_phase = phase
        
        # --- Local Oscillator Modulation Parameters ---
        mod_enabled = self.lo_params.get('mod_enabled', False)
        phase_delta = self.lo_params.get('phase_delta', 0.08)
        phase_freq = self.lo_params.get('phase_freq', 0.1)
        phase_offset = self.lo_params.get('phase_offset', 0.0)
        
        # --- Set Local Oscillator Nonidealities ---
        freq_drift_ppm = self.lo_params.get('freq_drift_ppm', 0.0)
        phase_noise_std = self.lo_params.get('phase_noise_std', 0.0)
        amp_noise_std = self.lo_params.get('amp_noise_std', 0.0)
        harmonic_distortion = self.lo_params.get('harmonic_distortion', 0.0)
        
        # --- Frequency drift (ppm → fractional offset) ---
        freq_drift = f0 * (freq_drift_ppm * 1e-6)

        # --- Compute pre-start time for edge case handling ---
        dt = real_time[1] - real_time[0]
        pre_start_time = real_time[0] - dt

        # --- Optional phase modulation ---
        if mod_enabled:
            phase_mod = (phase_delta / phase_freq) * \
                        np.sin(2 * np.pi * phase_freq * real_time + phase_offset)
            pre_phase_mod = (phase_delta / phase_freq) * \
                            np.sin(2 * np.pi * phase_freq * pre_start_time + phase_offset)
        else:
            phase_mod = 0.0
            pre_phase_mod = 0.0

        # --- Phase noise (white, Gaussian) ---
        phase_noise = self.rng.normal(0, phase_noise_std, len(real_time))
        pre_phase_noise = self.rng.normal(0, phase_noise_std)  # single value

        # --- Amplitude noise (white, Gaussian) ---
        amp_noise = self.rng.normal(0, amp_noise_std, len(real_time))
        pre_amp_noise = self.rng.normal(0, amp_noise_std)
        amp = A * (1 + amp_noise)
        pre_amp = A * (1 + pre_amp_noise)

        # --- Instantaneous phase (includes drift, modulation, and noise) ---
        phase = 2 * np.pi * (f0 + freq_drift) * real_time + \
                phase + phase_mod + phase_noise

        pre_phase = 2 * np.pi * (f0 + freq_drift) * pre_start_time + \
                    pre_phase + pre_phase_mod + pre_phase_noise

        # --- Fundamental signal ---
        lo = amp * np.sin(phase)
        pre_start_lo = pre_amp * np.sin(pre_phase)

        # --- Add 2nd harmonic distortion ---
        harmonic_amp = harmonic_distortion * A
        if harmonic_amp != 0:
            lo += harmonic_amp * np.sin(2 * phase)
            pre_start_lo += harmonic_amp * np.sin(2 * pre_phase)
            
        results = [lo]

        if return_phase_mod:
            results.append(phase_mod)
        if return_pre_start:
            results.append(pre_start_lo)
        if return_effects:
            effects = LOEffects(phase_noise=phase_noise,
                                pre_phase_noise=pre_phase_noise,
                                amp_noise=amp_noise,
                                pre_amp_noise=pre_amp_noise)
            results.append(effects)

        return tuple(results)

    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_config_name(self):
        return self.config_name
    
    
    def get_lo_params(self):
        return self.lo_params
    
    
    def get_log_params(self):
        return self.log_params