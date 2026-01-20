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


@dataclass(frozen=True)
class LOResult:
    lo: np.ndarray
    phase_mod: np.ndarray | None = None
    pre_start_lo: float | None = None
    effects: LOEffects | None = None
    

class LocalOscillator:
    """
    Represents a realistic local oscillator (LO) with optional modulation, noise,
    frequency drift, and harmonic distortion.
    """

    def __init__(self,
                 all_params=None,
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
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "lo_params": lo_params,
                "config_name": config_name,
                "log_params": log_params
            }

        self.set_all_params(all_params)
 
    # -------------------------------
    # Setters
    # -------------------------------
       
    def set_all_params(self, all_params=None):
        if all_params is None:
            all_params = {}
        
        lo_params = all_params.get('lo_params', None)
        log_params = all_params.get('log_params', None)
        if lo_params is None:
            config_name = "Default_LO_Config"
        else:
            config_name = all_params.get('config_name', "LO_Config_1")
        
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
        
        self.set_lo_params(lo_params)
        self.set_config_name(config_name)

        
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

    def generate_signal(
        self,
        real_time: np.ndarray,
        return_phase_mod: bool = False,
        return_pre_start: bool = False,
        return_effects: bool = False
    ) -> LOResult:
        # --- Parameters ---
        f0 = self.lo_params.get('freq', 100)
        A = self.lo_params.get('amp', 1)
        phase0 = self.lo_params.get('phase', 0)

        mod_enabled = self.lo_params.get('mod_enabled', False)
        phase_delta = self.lo_params.get('phase_delta', 0.08)
        phase_freq = self.lo_params.get('phase_freq', 0.1)
        phase_offset = self.lo_params.get('phase_offset', 0.0)

        freq_drift_ppm = self.lo_params.get('freq_drift_ppm', 0.0)
        phase_noise_std = self.lo_params.get('phase_noise_std', 0.0)
        amp_noise_std = self.lo_params.get('amp_noise_std', 0.0)
        harmonic_distortion = self.lo_params.get('harmonic_distortion', 0.0)

        # --- Time ---
        dt = real_time[1] - real_time[0]
        pre_start_time = real_time[0] - dt

        # --- Frequency drift ---
        freq_drift = f0 * freq_drift_ppm * 1e-6

        # --- Phase modulation ---
        if mod_enabled:
            phase_mod = (phase_delta / phase_freq) * np.sin(
                2 * np.pi * phase_freq * real_time + phase_offset
            )
            pre_phase_mod = (phase_delta / phase_freq) * np.sin(
                2 * np.pi * phase_freq * pre_start_time + phase_offset
            )
        else:
            phase_mod = None
            pre_phase_mod = 0.0

        # --- Noise ---
        phase_noise = self.rng.normal(0, phase_noise_std, len(real_time))
        pre_phase_noise = self.rng.normal(0, phase_noise_std)

        amp_noise = self.rng.normal(0, amp_noise_std, len(real_time))
        pre_amp_noise = self.rng.normal(0, amp_noise_std)

        amp = A * (1 + amp_noise)
        pre_amp = A * (1 + pre_amp_noise)

        # --- Phase ---
        phase = (
            2 * np.pi * (f0 + freq_drift) * real_time
            + phase0
            + (phase_mod if phase_mod is not None else 0.0)
            + phase_noise
        )

        pre_phase = (
            2 * np.pi * (f0 + freq_drift) * pre_start_time
            + phase0
            + pre_phase_mod
            + pre_phase_noise
        )

        # --- Signal ---
        lo = amp * np.sin(phase)
        pre_start_lo = pre_amp * np.sin(pre_phase)

        # --- Harmonic distortion ---
        if harmonic_distortion != 0:
            lo += harmonic_distortion * A * np.sin(2 * phase)
            pre_start_lo += harmonic_distortion * A * np.sin(2 * pre_phase)

        # --- Effects ---
        effects = None
        if return_effects:
            effects = LOEffects(
                phase_noise=phase_noise,
                pre_phase_noise=pre_phase_noise,
                amp_noise=amp_noise,
                pre_amp_noise=pre_amp_noise
            )

        return LOResult(
            lo=lo,
            phase_mod=phase_mod if return_phase_mod else None,
            pre_start_lo=pre_start_lo if return_pre_start else None,
            effects=effects
        )

    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_config_name(self):
        return self.config_name
    
    
    def get_lo_params(self):
        return self.lo_params
    
    
    def get_log_params(self):
        return self.log_params
    

    def get_all_params(self):
        all_params = {
            "lo_params": self.lo_params,
            "config_name": self.config_name,
            "log_params": self.log_params
        }
        return all_params