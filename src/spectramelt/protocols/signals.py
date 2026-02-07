from typing import(
    Protocol,
    TypedDict,
    Tuple,
    List,
    Dict,
    Any,
    Literal,
    # NotRequired,
    TYPE_CHECKING
)
# Use typing_extensions.NotRequired for backward compatibility with Python <3.11
from typing_extensions import NotRequired
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from keras import Model
else:
    Model = object # runtime placeholder

class SignalSet(TypedDict):
    time_path: Path
    freq_path: Path
    wave_path: Path
    time_set: List
    wave_set: List


class AllInputSetSignals(TypedDict):
    dataset: SignalSet
    recovery: SignalSet


class InputSetParams(TypedDict):
    normalize: NotRequired[bool]
    overwrite: NotRequired[bool]
    fft_shift: NotRequired[bool]
    num_sigs: NotRequired[int]
    num_recovery_sigs: NotRequired[int]
    tones_per_sig: NotRequired[list[int]]
    wave_precision: NotRequired[int]
    
    
class WaveParams(TypedDict, total=False):
    amp: float
    freq: float
    phase: float
    real: float
    imag: float
    
    
class WavesParams(TypedDict, total=False):
    freq_range: NotRequired[Tuple[int, int]]
    amp_range: NotRequired[Tuple[float, float]]
    phase_range: NotRequired[Tuple[float, float]]
    waves: NotRequired[List[Dict[str, Any]]]
         
    
class InputSetAllParams(TypedDict):
    wave_params: NotRequired[WavesParams]


class InputSignalResultParams(TypedDict):
    input_signal: NotRequired[np.ndarray]


class InputSignalProtocol(Protocol):
    def get_inputset_params(self) -> InputSetParams: ...
    def get_freq_modes(self) -> list[str]: ...
    def get_config_name(self) -> str: ...
    def get_all_params(self) -> InputSetAllParams: ...
    def set_wave_params(self, wave_params: WavesParams) -> None: ...
    def create_input_signal(self, real_time: np.ndarray) -> InputSignalResultParams: ...
    
    
class AnalogDataProtocol(Protocol):
    """Protocol for objects providing analog time-series data."""

    @property
    def time(self) -> np.ndarray: ...
    
    @property
    def frequency(self) -> np.ndarray: ...  
    

class AnalogProtocol(Protocol):
    def create_analog(self) -> AnalogDataProtocol: ...
 

class PreMultiplyParams(TypedDict):
    normalize: NotRequired[bool]
    apply_fft: NotRequired[bool]
    overwrite: NotRequired[bool]
    fft_shift: NotRequired[bool]
    scale_dict: NotRequired[float]
    
    
class NormParams(TypedDict):
    input_type: NotRequired[str]
    output_type: NotRequired[str]
    

class TrainingParams(TypedDict):
    norm_params: NotRequired[NormParams]
    

class MLPProtocol(Protocol):
    def reset_tensorflow_session(self) -> None: ...
    def get_training_params(self) -> TrainingParams: ...
    def load_model(self, model_file_path: Path | None) -> Model: ...
    def set_recovery_stats_from_h5(
        self,
        norm_h5_path: Path,
        dataset_name: str
    ) -> None: ...
    def model_prediction(
        self,
        init_guess: np.ndarray,
        mode: Literal[
            "complex", "real", "imag",
            "real_imag", "mag", "ang",
            "mag_ang", "mag_ang_sincos"
        ]
    ) -> np.ndarray: ...

    def set_model_file_path(
        self,
        model_file_path: str
    ) -> None: ...


class RecoveryAllParams(TypedDict):
    pass


class RecoveryParams(TypedDict):
    method: str
    
    
class DataframeParams(TypedDict):
    file_path: NotRequired[Path]


class RecoveryProtocol(Protocol):
    def process_signal_file(self,
                            recovery_file: Path,
                            wbf_wave_file: Path,
                            input_file: Path,
                            num_recovery_sigs: int,
                            dataset_config_name: str,
                            inputset_config_name: str,
                            DUT_config_name: str,
                            recovery_config_name: str,
                            wbf_freq: np.ndarray) -> list[dict]: ...
    def create_rec_df(self,
                      inputset_config_file: Path | None,
                      recovery_df_file_path: Path | None) -> None: ...
    def get_dataframe_params(self) -> DataframeParams: ...
    def set_recovery_type(self, recovery_type: str) -> None: ...
    def get_freq_modes(self) -> list[str]: ...
    def get_premultiply_params(self) -> PreMultiplyParams: ...
    def get_config_name(self) -> str: ...
    def get_all_params(self) -> RecoveryAllParams: ...
    def get_recovery_params(self) -> RecoveryParams: ...
    def recover_signal(
        self,
        signal: np.ndarray,
        dictionary: np.ndarray | None = None,
        *,
        mlp: MLPProtocol | None = None
    ) -> np.ndarray: ...