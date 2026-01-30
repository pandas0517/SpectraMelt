import numpy as np
from .results import MixerResult
from .utils import load_config_from_json, get_logger


class Mixer:
    def __init__(self,
                 all_params=None,
                 mixer_params=None,
                 log_params=None,
                 config_name=None,
                 config_file_path=None) -> None:
        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "mixer_params": mixer_params,
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

        mixer_params = all_params.get('mixer_params', None)
        log_params = all_params.get('log_params', None)
        config_name = all_params.get('config_name', "Mixer_Config_1")

        self.set_log_params(log_params)

        self.logger = None
        if self.log_params.get('enabled', True):
            self.logger = get_logger(
                self.__class__.__name__,
                self.log_params.get('log_file', None),
                self.log_params.get('level', "INFO"),
                self.log_params.get('console', True)
            )

        self.set_mixer_params(mixer_params)
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

    def mix(self, rf_signal: np.ndarray, lo_signal: np.ndarray, return_effects=False) -> MixerResult:
        conversion_gain = self.mixer_params.get('conversion_gain', 1.0)
        lo_leakage = self.mixer_params.get('lo_leakage', 0.0)
        rf_leakage = self.mixer_params.get('rf_leakage', 0.0)
        nonlinearity_coeff = self.mixer_params.get('nonlinearity_coeff', 0.0)
        noise_std = self.mixer_params.get('noise_std', 0.0)

        mixed = conversion_gain * rf_signal * lo_signal
        mixed += lo_leakage * lo_signal
        mixed += rf_leakage * rf_signal
        mixed += nonlinearity_coeff * (rf_signal ** 3)

        noise = None
        if noise_std > 0:
            noise = self.rng.normal(0, noise_std, size=mixed.shape)
            mixed += noise

        return MixerResult(
            mixed=mixed,
            noise=noise if return_effects else None
        )

    # -------------------------------
    # Getters
    # -------------------------------

    def get_config_name(self):
        return self.config_name

    def get_mixer_params(self):
        return self.mixer_params.copy()

    def get_log_params(self):
        return self.log_params.copy()

    def get_all_params(self):
        return {
            "mixer_params": self.mixer_params.copy(),
            "log_params": self.log_params.copy(),
            "config_name": self.config_name
        }