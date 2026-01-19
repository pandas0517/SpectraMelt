import numpy as np
from dataclasses import dataclass
from .utils import load_config_from_json, get_logger
from typing import Optional


@dataclass(frozen=True)
class PGEffects:
    jitter: np.ndarray | None = None
    amp_noise: list[float] | None = None


@dataclass(frozen=True)
class PGResult:
    pulses: np.ndarray
    effects: Optional[PGEffects] | None = None


class PulseGenerator:
    def __init__(self,
                 all_params=None,
                 pulse_params=None,
                 log_params=None,
                 config_name=None,
                 config_file_path=None) -> None:

        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "pulse_params": pulse_params,
                "config_name": config_name,
                "log_params": log_params
            }

        self.set_all_params(all_params)

    # -------------------------------
    # Setters
    # -------------------------------

    def set_all_params(self, all_params):
        if all_params is None:
            all_params = {}

        pulse_params = all_params.get('pulse_params', None)
        log_params = all_params.get('log_params', None)
        if pulse_params is None:
            config_name = "Default_Pulse_Config"
        else:
            config_name = all_params.get('config_name', "Pulse_Config_1")

        self.set_log_params(log_params)
        self.logger = None
        if self.log_params.get('enabled', True):
            self.logger = get_logger(
                self.__class__.__name__,
                self.log_params.get('log_file', None),
                self.log_params.get('level', "INFO"),
                self.log_params.get('console', True)
            )

        self.set_pulse_params(pulse_params)
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

    def generate(self, signal, real_time, pre_start_val=None,
                 return_effects=False) -> PGResult:

        pulse_width = self.pulse_params.get('pulse_width', None)
        amplitude = self.pulse_params.get('amplitude', 1.0)
        rise_time = self.pulse_params.get('rise_time', 0.0)
        fall_time = self.pulse_params.get('fall_time', 0.0)
        baseline_offset = self.pulse_params.get('baseline_offset', 0.0)

        if pulse_width is None:
            pulse_width = real_time[1] - real_time[0]

        jitter_std = self.pulse_params.get('jitter_std', 0.0)
        amp_noise_std = self.pulse_params.get('amp_noise_std', 0.0)
        droop_coeff = self.pulse_params.get('droop_coeff', 0.0)

        ideal_pulses = self._rising_zero_crossings(signal, pre_start_val)
        pulse_times = real_time[ideal_pulses != 0]
        num_pulses = len(pulse_times)

        jitter = None
        if jitter_std > 0:
            jitter = self.rng.normal(0, jitter_std, num_pulses)
            pulse_times = pulse_times + jitter

        pulses = np.zeros_like(real_time)
        amp_noise_list = []

        for t0 in pulse_times:
            start = t0
            end = t0 + pulse_width

            if end <= real_time[0] or start >= real_time[-1]:
                continue

            start_idx = np.searchsorted(real_time, start, side='left')
            end_idx = np.searchsorted(real_time, end, side='right')

            amp = amplitude
            if amp_noise_std > 0:
                amp_noise = self.rng.normal(0, amp_noise_std)
                amp *= 1 + amp_noise
                if return_effects:
                    amp_noise_list.append(amp_noise)

            if droop_coeff > 0:
                amp *= np.exp(-droop_coeff * t0)

            if rise_time > 0:
                rise_end_idx = np.searchsorted(real_time, start + rise_time, side='right')
                pulses[start_idx:rise_end_idx] += amp * (real_time[start_idx:rise_end_idx] - start) / rise_time

            if fall_time > 0:
                fall_start_idx = np.searchsorted(real_time, end - fall_time, side='left')
                pulses[fall_start_idx:end_idx] += amp * (1 - (real_time[fall_start_idx:end_idx] - (end - fall_time)) / fall_time)

            flat_start_idx = start_idx + int(rise_time / (real_time[1] - real_time[0])) if rise_time > 0 else start_idx
            flat_end_idx = end_idx - int(fall_time / (real_time[1] - real_time[0])) if fall_time > 0 else end_idx

            if flat_end_idx > flat_start_idx:
                pulses[flat_start_idx:flat_end_idx] += amp

        pulses += baseline_offset

        effects = None
        if return_effects:
            effects = PGEffects(
                jitter=jitter,
                amp_noise=amp_noise_list
            )

        return PGResult(pulses=pulses, effects=effects)

    def _rising_zero_crossings(self, signal, pre_start_val=None):
        pulses = np.zeros_like(signal)

        if pre_start_val is not None:
            if np.sign(pre_start_val) < 0 and np.sign(signal[0]) >= 0:
                pulses[0] = 1

        signs = np.sign(signal)
        signs[signs == 0] = 1
        zero_crossings = np.where(np.diff(signs))[0] + 1

        for i in zero_crossings:
            if signal[i] > signal[i - 1]:
                pulses[i] = 1

        return pulses

    # -------------------------------
    # Getters
    # -------------------------------

    def get_config_name(self):
        return self.config_name

    def get_pulse_params(self):
        return self.pulse_params

    def get_log_params(self):
        return self.log_params

    def get_all_params(self):
        return {
            "pulse_params": self.pulse_params,
            "config_name": self.config_name,
            "log_params": self.log_params
        }