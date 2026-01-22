# Expose selected functions/classes from submodules

# From components.py
from .components import (
    ADCEffects,
    ADCResult,
    LPFResult,
    LOEffects,
    LOResult,
    PGEffects,
    PGResult,
    MixerResult,
    WaveletEffects,
    WaveletResult
)

# From device.py
from .device import (
    WBFResult,
    NYFRDictionary,
    NYFRResult,
    NFWBSDictionary,
    NFWBSResult
)

# From signals.py
from .signals import (
    AnalogData,
    InputSignalEffects,
    InputSignalResult,
    ConditionedSignal,
    SampleHoldSignal,
    QuantizedSignal
)


# Optional: define __all__ for clean "from utils import *"
__all__ = [
    "ConditionedSignal",
    "SampleHoldSignal",
    "QuantizedSignal",
    "ADCEffects",
    "ADCResult",
    "LPFResult",
    "LOEffects",
    "LOResult",
    "PGEffects",
    "PGResult",
    "MixerResult",
    "WaveletEffects",
    "WaveletResult",
    "WBFResult",
    "NYFRDictionary",
    "NYFRResult",
    "NFWBSDictionary",
    "NFWBSResult",
    "AnalogData",
    "InputSignalEffects",
    "InputSignalResult",
]