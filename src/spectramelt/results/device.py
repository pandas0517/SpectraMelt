from dataclasses import dataclass
import numpy as np
from .components import(
    LPFResult,
    ADCResult,
    LOResult,
    PGResult,
    MixerResult,
    WaveletResult
)

# ============================================================
# === Wide Band Filtered Signal & Result Containers
# ============================================================

@dataclass(frozen=True)
class WBFResult:
    wbf_signal: LPFResult | None = None
    wbf_sub_sig: np.ndarray | None = None
    time: np.ndarray | None = None
    freq: np.ndarray | None = None

# ============================================================
# === NYFR Dictionary & Result Containers
# ============================================================

@dataclass(frozen=True)
class NYFRResult:
    adc_signal: ADCResult | None = None
    wbf_signal: WBFResult | None = None
    lo_signal: LOResult | None = None
    pulse_signal: PGResult | None = None
    mixed_signal: MixerResult | None = None
    lpf_signal: LPFResult | None = None
    lo_phase_mod_mid: np.ndarray | None = None


@dataclass(frozen=True)
class NYFRDictionary:
    dictionary: np.ndarray | None = None
    zones: int | None = None
    k_bands: int | None = None
    
# ============================================================
# === NFWBS Dictionary & Result Containers
# ============================================================    

@dataclass(frozen=True)
class NFWBSResult:
    adc_signal: ADCResult | None = None
    wbf_signal: WBFResult | None = None
    lo_1_signal: LOResult | None = None
    pulse_1_signal: PGResult | None = None
    mixed_1_signal: MixerResult | None = None
    lpf_1_signal: LPFResult | None = None
    lo_2_signal: LOResult | None = None
    pulse_2_signal: PGResult | None = None
    wavelet_signal: WaveletResult | None = None
    mixed_2_signal: MixerResult | None = None
    lpf_2_signal: LPFResult | None = None
    lo_phase_mod_mid: np.ndarray | None = None



@dataclass(frozen=True)
class NFWBSDictionary:
    dictionary: np.ndarray | None = None
    zones: int | None = None
    k_bands: int | None = None