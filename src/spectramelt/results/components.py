from dataclasses import dataclass
import numpy as np
from typing import Optional, List
from .signals import (
    ConditionedSignal,
    SampleHoldSignal,
    QuantizedSignal
)

# ============================================================
# === ADC Signal & Result Containers
# ============================================================

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
# === Low Pass Filter Signal & Result Containers
# ============================================================

@dataclass(frozen=True)
class LPFResult:
    filtered: np.ndarray
    noise: np.ndarray | None = None
    
# ============================================================
# === Local Oscillator Signal & Result Containers
# ============================================================
    
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
    
# ============================================================
# === Pulse Generator Signal & Result Containers
# ============================================================

@dataclass(frozen=True)
class PGEffects:
    jitter: np.ndarray | None = None
    amp_noise: list[float] | None = None


@dataclass(frozen=True)
class PGResult:
    pulses: np.ndarray
    effects: Optional[PGEffects] | None = None
    
# ============================================================
# === Mixer Signal & Result Containers
# ============================================================

@dataclass(frozen=True)
class MixerResult:
    mixed: np.ndarray
    noise: np.ndarray | None = None
    
# ============================================================
# === Wavelet Generator Signal & Result Containers
# ============================================================

@dataclass(frozen=True)
class WaveletEffects:
    amp_noise: np.ndarray | None = None
    drift: float | None = None


@dataclass(frozen=True)
class WaveletResult:
    wavelet_train: np.ndarray | None = None
    components: list[np.ndarray] | None = None
    effects: Optional[WaveletEffects] | None = None
    amp: float | None = None