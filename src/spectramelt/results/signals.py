from dataclasses import dataclass
import numpy as np

# ============================================================
# === Analog Simulator Data Container
# ============================================================

@dataclass(frozen=True)
class AnalogData:
    sim_freq: float  | None = None         # points per second
    adj_spacing: float | None = None       # spacing between points (1/fs)
    total_time: float | None = None        # duration of the signal
    num_points: int | None = None          # number of samples
    time: np.ndarray | None = None         # time vector
    frequency: np.ndarray | None = None    # frequency vector

# ============================================================
# === Input Signal & Result Containers
# ============================================================

@dataclass(frozen=True)
class InputSignalEffects:
    noise: np.ndarray | None = None
    delay: tuple[float, ...] | None = None
    echo_att: tuple[float, ...] | None = None
    local_doppler: tuple[float, ...] | None = None
    phase_inversion: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class InputSignalResult:
    input_signal: np.ndarray | None = None
    effects: InputSignalEffects | None = None
    
# ============================================================
# === ADC Signal Containers
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