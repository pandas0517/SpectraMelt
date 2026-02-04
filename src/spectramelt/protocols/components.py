from typing import(
    Protocol,
    TypedDict
)
import numpy as np


class QuantizedSignalProtocol(Protocol):
    
    @property
    def quantized_values(self) -> np.ndarray: ...
    
    @property
    def mid_times(self) -> np.ndarray: ...
    
    @property
    def sampled_frequency(self) -> np.ndarray: ...


class ADCResultProtocol(Protocol):
    
    @property
    def quantized(self) -> QuantizedSignalProtocol: ...
    
    
class WBFResultProtocol(Protocol):
    
    @property
    def wbf_sub_sig(self) -> np.ndarray: ...
    
    @property
    def time(self) -> np.ndarray: ...
    
    @property
    def freq(self) -> np.ndarray: ...
    

class LOParams(TypedDict):
    freq: int