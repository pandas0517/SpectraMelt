from typing import Protocol, Literal
import numpy as np

class RecoveryMLPProtocol(Protocol):
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