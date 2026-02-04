from typing import(
    Protocol,
    TypedDict,
    NotRequired
)
from .components import (
    ADCResultProtocol,
    WBFResultProtocol,
    LOParams
)
import numpy as np
  

class OutputSetParams(TypedDict):
    normalize: NotRequired[bool]
    fft_shift: NotRequired[bool]
    normalize_wbf: NotRequired[bool]
    fft_shift_wbf: NotRequired[bool]
    overwrite: NotRequired[bool]
    DUT_type: NotRequired[str]
    

class OutputSetAllParams(TypedDict):
    pass


class OutputSetFreqModes(TypedDict):
    output: list[str]
    wideband: list[str]


class OutputSignalResultProtocol(Protocol):
    
    @property
    def adc_signal(self) -> ADCResultProtocol: ...
    
    @property
    def wbf_signal(self) -> WBFResultProtocol: ...
    
    @property
    def lo_phase_mod_mid(self) -> np.ndarray: ...
    
    
class OutputSignalDictProtocol(Protocol):
    
    @property
    def dictionary(self) -> np.ndarray: ...


class DUTProtocol(Protocol):
    def get_outputset_params(self) -> OutputSetParams: ...
    def get_lo_params(self) -> LOParams: ...
    def get_config_name(self) -> str: ...
    def get_all_params(self) -> OutputSetAllParams: ...
    def get_freq_modes(self) -> OutputSetFreqModes: ...
    def create_output_signal(self, input_signal: np.ndarray, real_time: np.ndarray) -> OutputSignalResultProtocol: ...
    def create_dictionary(self, lo_phase_mod_mid: np.ndarray, wbf_time: np.ndarray) -> OutputSignalDictProtocol: ...