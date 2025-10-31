import numpy as np
import os
from importlib import import_module
from .utils import load_config_from_json, get_logger

class Recovery:

    def __init__(self,
                signal=None,
                dictionary=None,
                recovery_params=None,
                log_params=None,
                config_name=None,
                num_waves=1,
                config_file_path=None) -> None:
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_recovery_params(recovery_params)
            if recovery_params is None:
                config_name = "Default_Recovery_Config"
            self.set_config_name(config_name)
            self.set_log_params(log_params)
            
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
            if config_file_path is not None:
                self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")
                
        self.recovered_coefs = None
        if signal is not None and dictionary is not None:
            self.recovered_coefs = self.recover_signal(signal, dictionary, num_waves)
    # -------------------------------
    # Setters
    # -------------------------------

    def set_config_from_file(self, config_file_path):
        recovery_config = load_config_from_json(config_file_path)
        recovery_params = recovery_config.get('recovery_params', None)
        log_params = recovery_config.get('log_params', None)
        config_name = recovery_config.get('config_name', "Recovery_Config_1")
        
        if recovery_params is None:
            config_name = "Default_Recovery_Config"

        self.set_log_params(log_params)
        self.set_recovery_params(recovery_params)
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

    def set_recovery_params(self, recovery_params=None):
        if recovery_params is None:
            recovery_params = {
                "method": "SPGL1",
                "threshold_frac": 0.05,
                "auto_threshold": False,
                "recovery_type": "complex",
                "model_file_path": None,
                "sigma": 0.001,
                "dict_mag_adj": 1.0
            }
        self.recovery_params = recovery_params

              
    # -------------------------------
    # Core functional methods
    # -------------------------------

    def _sparse_fft(self, signal):
        """
        Compute sparse magnitude and phase of a time-domain signal using time vector.

        Parameters
        ----------
        signal : array_like
            Input time-domain signal (real or complex)
        threshold_frac : float, optional
            Keep bins with magnitude > threshold_frac * max(magnitude)
            Default = 0.05 (5%). Ignored if auto_threshold=True
        auto_threshold : bool, optional
            If True, automatically determine threshold using median + 2*std method

        Returns
        -------
        magnitude_sparse : ndarray
            Magnitude spectrum with zeroed values below threshold
        phase_sparse : ndarray
            Phase spectrum with zeroed values below threshold
        mask : ndarray
            Boolean mask of where significant tones are kept
        """                      
        threshold_frac = self.recovery_params.get('threshold_frac', 0.05)
        auto_threshold = self.recovery_params.get('auto_threshold', False)

        fft_vals = np.fft.fft(signal)

        # Magnitude + Phase
        magnitude = np.abs(fft_vals)
        phase = np.angle(fft_vals)

        # Determine threshold
        if auto_threshold:
            # Automatic threshold: median + 2*std (adjust factor as needed)
            threshold = np.median(magnitude) + 2*np.std(magnitude)
        else:
            threshold = np.max(magnitude) * threshold_frac

        # Significant bin mask
        mask = magnitude > threshold

        # Zero output outside mask
        magnitude_sparse = magnitude * mask
        phase_sparse = phase * mask

        return magnitude_sparse, phase_sparse, mask


    def recover_signal(self, signal, dictionary, num_waves=1):
        sigma = self.recovery_params.get('sigma', 0.001)
        dict_mag_adj = self.recovery_params.get('dict_mag_adj', 1.0)
        model_file_path = self.recovery_params.get('model_file_path', None)
        recovery_method = self.recovery_params.get('method', "splg1").lower()
        recovery_type = self.recovery_params.get('recovery_type', "complex").lower()

        recovered_coef = None
        sparse_signal = None

        magnitude_sparse, phase_sparse, _ = self._sparse_fft(signal)
        complex_sparse = magnitude_sparse * np.exp(1j * phase_sparse)

        match recovery_type:
            case 'complex':
                if np.iscomplexobj(dictionary):
                        if recovery_method == "mlp1":
                            sparse_signal = np.concatenate([magnitude_sparse, phase_sparse])
                        else:
                            sparse_signal = complex_sparse
                else:
                    self.logger.error("Dictionary is not complex")
                    raise ValueError("Dictionary is not complex")
            case 'real':
                sparse_signal = complex_sparse.real
            case 'imag':
                if np.iscomplexobj(dictionary):
                        sparse_signal = complex_sparse.imag
                else:
                    self.logger.error("Dictionary is not complex")
                    raise ValueError("Dictionary is not complex")
            case 'mag':
                sparse_signal = magnitude_sparse
            case 'phase':
                if np.iscomplexobj(dictionary):
                        sparse_signal = phase_sparse
                else:
                    self.logger.error("Dictionary is not complex")
                    raise ValueError("Dictionary is not complex")

        match recovery_method:
            case 'iht':
                IHT = import_module("IHT")
                recovered_coef = IHT.CIHT(dict_mag_adj * dictionary, sparse_signal, 2*num_waves, learning_rate=sigma)
            case 'omp':
                OMP = import_module("OMP")
                signal_norm = np.linalg.norm(sparse_signal)
                recovered_coef = OMP.OMP(dict_mag_adj * dictionary, sparse_signal/signal_norm)[0]
            case 'spgl1':
                spgl1 = import_module("spgl1")
                signal_norm = np.linalg.norm(sparse_signal)
                recovered_coef,_,_,_ = spgl1.spgl1(dict_mag_adj * dictionary, sparse_signal/signal_norm, sigma=sigma)
            case 'mlp1':
                NYFR_ML_Models = import_module("NYFR_ML_Models")
                if not os.path.exists(model_file_path):
                    raise FileNotFoundError(f"File not found: {model_file_path}")
                pseudo = np.linalg.pinv(dict_mag_adj *dictionary)
                init_guess = np.dot(pseudo,sparse_signal)
                recovered_coef = NYFR_ML_Models.model_prediction(init_guess, model_file_path)
            case _:
                self.logger.error(f"Recovery method {recovery_method} is not supported")

        return recovered_coef

    # -------------------------------
    # Getters
    # -------------------------------

    def get_config_name(self):
        return self.config_name
    
    def get_recovered_coefs(self):
        return self.recovered_coefs
    
    def get_recovery_params(self):
        return self.recovery_params

    def get_log_params(self):
        return self.log_params