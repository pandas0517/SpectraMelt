import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .utils import load_config_from_json, get_logger


# ============================================================
# === Signal & Result Containers
# ============================================================

@dataclass(frozen=True)
class ConditionedSignal:
    signal: np.ndarray
    time: np.ndarray
    freq: np.ndarray
    total_time: float


@dataclass(frozen=True)
class SampleHoldSignal:
    output_signal: np.ndarray
    indices: np.ndarray
    sampled_values: np.ndarray


@dataclass(frozen=True)
class QuantizedSignal:
    quantized_values: np.ndarray
    mid_times: np.ndarray
    adc_indices: np.ndarray
    sampled_frequency: np.ndarray


@dataclass(frozen=True)
class ADCEffects:
    jitter_indices: Optional[np.ndarray] = None
    hold_noise: Optional[List[float]] = None
    thermal_noise: Optional[List[float]] = None


@dataclass(frozen=True)
class ADCResult:
    quantized: QuantizedSignal
    conditioned: Optional[ConditionedSignal] = None
    sample_hold: Optional[SampleHoldSignal] = None
    effects: Optional[ADCEffects] = None


# ============================================================
# === ADC Core
# ============================================================

class ADC:
    def __init__(
        self,
        all_params=None,
        adc_params=None,
        log_params=None,
        config_name=None,
        config_file_path=None
    ) -> None:

        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "adc_params": adc_params,
                "config_name": config_name,
                "log_params": log_params
            }

        self.set_all_params(all_params)

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    def set_all_params(self, all_params):
        if all_params is None:
            all_params = {}

        adc_params = all_params.get("adc_params")
        log_params = all_params.get("log_params")
        config_name = all_params.get("config_name", "ADC_Config_1")

        self.set_log_params(log_params)
        self._init_logger()
        self.set_adc_params(adc_params)
        self.config_name = config_name

    def set_log_params(self, log_params=None):
        self.log_params = log_params or {
            "enabled": True,
            "log_file": None,
            "level": "INFO",
            "console": True
        }

    def _init_logger(self):
        self.logger = None
        if self.log_params.get("enabled", True):
            self.logger = get_logger(
                self.__class__.__name__,
                self.log_params.get("log_file"),
                self.log_params.get("level", "INFO"),
                self.log_params.get("console", True)
            )

    def set_adc_params(self, adc_params=None):
        self.adc_params = adc_params or {
            "adc_samp_freq": 100,
            "allow_clipping": True,
            "v_ref_range": (0.0, 1.0),
            "num_bits": 8,
            "thermal_noise_std_dev": 0.0,
            "non_linearity_mode": None,
            "alpha": 0.0,
            "threshold": 1.0,
            "jitter_std": 0.0,
            "acquisition_time_constant": 0.0,
            "hold_noise_std": 0.0,
            "transient_mode": "fixed",
            "truncate_transients": True,
            "transient_fraction": 0.05,
            "detection_window": 0.05,
            "stability_threshold": 0.01,
            "seed": None
        }
        self.rng = np.random.default_rng(self.adc_params.get("seed"))

    # ============================================================
    # === Processing Stages
    # ============================================================

    def _condition_adc_input(
        self,
        signal: np.ndarray,
        real_time: np.ndarray
    ) -> ConditionedSignal:

        n = signal.size
        if n == 0:
            raise ValueError("Empty signal provided to ADC")

        v_min, v_max = self.adc_params["v_ref_range"]
        transient_mode = self.adc_params["transient_mode"].lower()
        frac = self.adc_params["transient_fraction"]
        truncate = self.adc_params["truncate_transients"]

        if transient_mode == "fixed":
            skip = int(n * frac)
            skip_start = skip_end = min(skip, n // 2)
        else:
            skip_start = skip_end = 0

        steady = signal[skip_start:n - skip_end] if skip_start < n - skip_end else signal
        centered = steady - np.mean(steady)
        max_amp = np.max(np.abs(centered)) or 1.0

        scale = ((v_max - v_min) / 2.0) / max_amp
        scaled = centered * scale + (v_max + v_min) / 2.0

        if truncate:
            out_signal = scaled
            out_time = real_time[skip_start:n - skip_end]
        else:
            out_signal = np.full_like(signal, (v_max + v_min) / 2.0)
            out_signal[skip_start:n - skip_end] = scaled
            out_time = real_time

        dt = out_time[1] - out_time[0]
        freq = np.linspace(-0.5 / dt, 0.5 / dt, out_time.size, endpoint=False)

        return ConditionedSignal(
            signal=out_signal,
            time=out_time,
            freq=freq,
            total_time=out_time[-1] - out_time[0]
        )

    # ------------------------------------------------------------

    def _sample_and_hold(
        self,
        conditioned: ConditionedSignal
    ) -> Tuple[SampleHoldSignal, np.ndarray, List[float]]:

        signal = conditioned.signal
        time = conditioned.time

        dt = np.mean(np.diff(time))
        sim_freq = 1.0 / dt
        adc_fs = self.adc_params["adc_samp_freq"]

        num_samples = int(len(signal) * adc_fs / sim_freq)
        ideal_idx = np.arange(num_samples) * (sim_freq / adc_fs)
        jitter = self.rng.normal(0, self.adc_params["jitter_std"] * sim_freq, num_samples)

        indices = np.clip(ideal_idx + jitter, 0, len(signal) - 1).astype(int)
        sampled = signal[indices]

        output = np.zeros_like(signal)
        hold_noise = []

        for i in range(num_samples):
            start = indices[i]
            end = indices[i + 1] if i < num_samples - 1 else len(signal)
            output[start:end] = sampled[i]

            if self.adc_params["hold_noise_std"] > 0:
                n = self.rng.normal(0, self.adc_params["hold_noise_std"])
                output[start:end] += n
                hold_noise.append(n)

        return (
            SampleHoldSignal(output, indices, sampled),
            jitter,
            hold_noise
        )

    # ------------------------------------------------------------

    def _quantizer(
        self,
        sh: SampleHoldSignal,
        time: np.ndarray
    ) -> Tuple[QuantizedSignal, List[float]]:

        vmin, vmax = self.adc_params["v_ref_range"]
        bits = self.adc_params["num_bits"]
        levels = 2 ** bits
        q_step = (vmax - vmin) / levels

        mid_times = []
        values = []
        indices = []
        noise = []

        for i, start in enumerate(sh.indices):
            end = sh.indices[i + 1] if i < len(sh.indices) - 1 else len(sh.output_signal)
            mid = start + (end - start) // 2
            val = sh.output_signal[mid]

            if self.adc_params["thermal_noise_std_dev"] > 0:
                n = self.rng.normal(0, self.adc_params["thermal_noise_std_dev"])
                val += n
                noise.append(n)

            val = np.clip(val, vmin, vmax)
            idx = int((val - vmin) // q_step)
            idx = np.clip(idx, 0, levels - 1)

            values.append(vmin + (idx + 0.5) * q_step)
            indices.append(idx)
            mid_times.append(time[mid])

        dt = mid_times[1] - mid_times[0]
        freq = np.linspace(-0.5 / dt, 0.5 / dt, len(mid_times), endpoint=False)

        return (
            QuantizedSignal(
                quantized_values=np.array(values),
                mid_times=np.array(mid_times),
                adc_indices=np.array(indices),
                sampled_frequency=freq
            ),
            noise
        )

    # ============================================================
    # === Public API
    # ============================================================

    def analog_to_digital(
        self,
        signal: np.ndarray,
        real_time: np.ndarray,
        return_conditioned=False,
        return_sample_hold=False,
        return_effects=False
    ) -> ADCResult:

        conditioned = self._condition_adc_input(signal, real_time)
        sh, jitter, hold_noise = self._sample_and_hold(conditioned)
        quantized, thermal_noise = self._quantizer(sh, conditioned.time)

        effects = None
        if return_effects:
            effects = ADCEffects(jitter, hold_noise, thermal_noise)

        return ADCResult(
            quantized=quantized,
            conditioned=conditioned if return_conditioned else None,
            sample_hold=sh if return_sample_hold else None,
            effects=effects
        )

    # ============================================================
    # === Getters
    # ============================================================

    def get_adc_params(self):
        return self.adc_params


    def get_log_params(self):
        return self.log_params


    def get_config_name(self):
        return self.config_name


    def get_all_params(self):
        all_params = { 
            "adc_params": self.adc_params,
            "config_name": self.config_name,
            "log_params": self.log_params
            }
        return all_params