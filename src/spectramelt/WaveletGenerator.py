import numpy as np
from dataclasses import dataclass
from typing import Optional
from .utils import load_config_from_json, get_logger


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


class WaveletGenerator:
    """
    Generates a realistic Gabor wavelet train aligned to a pulse train for NFWBS / NYFR.
    Models amplitude noise, frequency drift, and harmonic distortion, emulating a real oscillator.
    """

    def __init__(self,
                 all_params=None,
                 wavelet_params=None,
                 log_params=None,
                 config_name=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------
        sample_train : np.ndarray
            Array of 0s and 1s indicating pulse positions (rising edges, ADC ticks)
        t : np.ndarray
            Time vector for waveform generation
        psi_params : list of dict
            Each dict has keys:
                'amp'   : amplitude
                'f_c'   : center frequency (Hz)
        realistic_params : dict
            Optional real-world imperfections:
                'amp_noise_std' : amplitude noise (fraction)
                'freq_drift_ppm' : frequency drift (ppm)
                'harmonic_distortion' : fractional 2nd harmonic
                'phase_noise_std' : phase noise in radians
                'seed' : random seed for reproducibility
        """
        # ---------------- Config ----------------
        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "wavelet_params": wavelet_params,
                "config_name": config_name,
                "log_params": log_params
            }

        self.set_all_params(all_params)

    # -------------------------------
    # Setters
    # -------------------------------

    def set_all_params(self, all_params):
        if all_params is None:
            all_params = {}
        
        wavelet_params = all_params.get('wavelet_params', None)
        log_params = all_params.get('log_params', None)
        if wavelet_params is None:
            config_name = "Default_Wavelet_Config"
        else:
            config_name = all_params.get('config_name', "Wavelet_Config_1")
        
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
        
        self.set_wavelet_params(wavelet_params)
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
        

    def set_wavelet_params(self, wavelet_params=None):
        if wavelet_params is None:
            wavelet_params = {
                "center_freq": 35,
                "amp_noise_std": 0.0,
                "freq_drift_ppm": 0.0,
                "harmonic_distortion": 0.0,
                "phase_noise_std": 0.0,
                "threshold": 1e-3,
                "seed": None
            }
        self.wavelet_params = wavelet_params
        self.rng = np.random.default_rng(wavelet_params.get('seed', None))

    # -------------------------------
    # Core functional methods
    # -------------------------------

    def generate_wavelet_train(self, sample_train, t,
                               return_components=False,
                               return_scaling_factor=False,
                               return_effects=False) -> WaveletResult:
        """
        Generate Gabor wavelet train (CPU). Automatically adds negative frequency.
        Scales to unit magnitude in frequency domain and stores required amplifier in self.amp.
        """
        # --- Defaults ---
        center_freq = self.wavelet_params['center_freq']
        amp_noise_std = self.wavelet_params.get('amp_noise_std', 0.0)
        freq_drift_ppm = self.wavelet_params.get('freq_drift_ppm', 0.0)
        harmonic_distortion = self.wavelet_params.get('harmonic_distortion', 0.0)
        threshold = self.wavelet_params.get('threshold', 1e-3)        
        
        dt = t[1] - t[0]
        wavelet_train = np.zeros_like(t, dtype=complex)
        components = []

        # --- Detect pulse regions ---
        active = sample_train > threshold
        edges = np.diff(active.astype(int))
        start_idxs = np.where(edges == 1)[0] + 1
        end_idxs = np.where(edges == -1)[0] + 1
        if active[0]:
            start_idxs = np.r_[0, start_idxs]
        if active[-1]:
            end_idxs = np.r_[end_idxs, len(sample_train)]

        # --- Process each pulse ---
        for s, e in zip(start_idxs, end_idxs):
            pulse = sample_train[s:e]
            tp = t[s:e]

            if np.sum(pulse) < threshold:
                continue

            # --- Pulse center and RMS width ---
            pulse_norm = pulse / np.sum(pulse)
            t0 = np.sum(tp * pulse_norm)
            tau = np.sqrt(np.sum(((tp - t0) ** 2) * pulse_norm))
            tau = max(tau, dt)

            # --- Wavelet normalization ---
            norm = (2 ** 0.25) / (np.sqrt(tau) * np.pi ** 0.25)

            # --- Fundamental wavelet with automatic negative frequency ---
            envelope = np.exp(-((t - t0) / tau) ** 2)
            carrier_pos = np.exp(2j * np.pi * center_freq * (t - t0))
            carrier_neg = np.exp(2j * np.pi * -center_freq * (t - t0))
            psi = norm * envelope * (carrier_pos + carrier_neg)

            wavelet_train += psi
            components.append(psi)

        # --- Compute scaling factor so |FFT| at center_freq = 1 ---
        W_f = np.fft.fft(wavelet_train) / len(t)  # FFT normalized by N
        f = np.fft.fftfreq(len(t), dt)
        idx = np.argmin(np.abs(f - center_freq))
        mag = np.abs(W_f[idx])
        amp = 1.0 / mag if mag > 0 else 1.0

        # --- Apply scaling ---
        wavelet_train *= amp
        components = [c * amp for c in components]

        # --- Apply realistic effects AFTER scaling ---
        amp_noise = None
        drift = None

        if amp_noise_std > 0:
            amp_noise = 1 + amp_noise_std * self.rng.standard_normal(wavelet_train.shape)
            wavelet_train *= amp_noise
        if freq_drift_ppm != 0 or harmonic_distortion != 0:
            drift = 1 + freq_drift_ppm * 1e-6 * self.rng.standard_normal()
            fc_drifted = center_freq * drift
            harmonic = np.exp(2j * np.pi * 2 * fc_drifted * t)
            wavelet_train += harmonic_distortion * amp * norm * envelope * harmonic

        effects = None
        if return_effects:
            effects = WaveletEffects(
                amp_noise=amp_noise,
                drift=drift
                )

        return WaveletResult(
            wavelet_train=wavelet_train,
            components=components if return_components else None,
            amp=amp if return_scaling_factor else None,
            effects=effects
        )
    
    
    def generate_wavelet_train_gpu(
        self,
        sample_train: np.ndarray,
        t: np.ndarray,
        return_components: bool = False,
        return_scaling_factor: bool = False,
        return_effects: bool = False
    ) -> WaveletResult:
        """
        Generate a Gabor wavelet train using GPU acceleration (CuPy).
        Automatically adds negative frequency and scales to unit magnitude.

        Parameters
        ----------
        sample_train : np.ndarray
            Pulse-shaped train
        t : np.ndarray
            Time vector
        amp_noise_std : float, optional
        freq_drift_ppm : float, optional
        harmonic_distortion : float, optional
        threshold : float, optional
        """
        import cupy as cp

        # --- Defaults ---
        center_freq = self.wavelet_params['center_freq']
        amp_noise_std = self.wavelet_params.get('amp_noise_std', 0.0)
        freq_drift_ppm = self.wavelet_params.get('freq_drift_ppm', 0.0)
        harmonic_distortion = self.wavelet_params.get('harmonic_distortion', 0.0)
        threshold = self.wavelet_params.get('threshold', 1e-3)

        dt = t[1] - t[0]

        # --- Move to GPU ---
        t_gpu = cp.asarray(t)
        sample_gpu = cp.asarray(sample_train, dtype=cp.float64)

        wavelet_train = cp.zeros_like(t_gpu, dtype=cp.complex128)
        components = []

        # --- Detect pulse regions ---
        active = sample_gpu > threshold
        edges = cp.diff(active.astype(cp.int32))
        start_idxs = cp.where(edges == 1)[0] + 1
        end_idxs = cp.where(edges == -1)[0] + 1

        if bool(active[0]):
            start_idxs = cp.concatenate([cp.array([0]), start_idxs])
        if bool(active[-1]):
            end_idxs = cp.concatenate([end_idxs, cp.array([len(sample_train)])])

        # --- Process each pulse ---
        for s, e in zip(start_idxs.tolist(), end_idxs.tolist()):
            pulse = sample_gpu[s:e]
            tp = t_gpu[s:e]

            if cp.sum(pulse) < threshold:
                continue

            # --- Pulse center and RMS width ---
            pulse_norm = pulse / cp.sum(pulse)
            t0 = cp.sum(tp * pulse_norm)
            tau = cp.sqrt(cp.sum(((tp - t0) ** 2) * pulse_norm))
            tau = cp.maximum(tau, dt)

            # --- Wavelet normalization ---
            norm = (2 ** 0.25) / (cp.sqrt(tau) * cp.pi ** 0.25)

            # --- Fundamental wavelet with negative frequency ---
            envelope = cp.exp(-((t_gpu - t0) / tau) ** 2)
            carrier_pos = cp.exp(2j * cp.pi * center_freq * (t_gpu - t0))
            carrier_neg = cp.exp(2j * cp.pi * -center_freq * (t_gpu - t0))
            psi = norm * envelope * (carrier_pos + carrier_neg)

            wavelet_train += psi
            components.append(psi)

        # --- Compute scaling factor so |FFT| at center_freq = 1 ---
        W_f = cp.fft.fft(wavelet_train) / len(t_gpu)
        f = cp.fft.fftfreq(len(t_gpu), dt)
        idx = int(cp.argmin(cp.abs(f - center_freq)))
        mag = cp.abs(W_f[idx])
        amp = float(1.0 / mag) if mag > 0 else 1.0

        # --- Apply scaling ---
        wavelet_train *= amp
        components = [c * amp for c in components]

        # --- Apply realistic effects AFTER scaling ---
        amp_noise = None
        drift = None

        if amp_noise_std > 0:
            amp_noise = 1 + amp_noise_std * cp.asarray(
                self.rng.standard_normal(wavelet_train.shape)
            )
            wavelet_train *= amp_noise

        if freq_drift_ppm != 0 or harmonic_distortion != 0:
            drift = 1 + freq_drift_ppm * 1e-6 * self.rng.standard_normal()
            fc_drifted = center_freq * drift
            harmonic = cp.exp(2j * cp.pi * 2 * fc_drifted * t_gpu)
            wavelet_train += harmonic_distortion * amp * norm * envelope * harmonic

        # --- Convert back to NumPy at boundary ---
        wavelet_train_np = cp.asnumpy(wavelet_train)
        components_np = [cp.asnumpy(c) for c in components] if return_components else None
        amp_noise_np = cp.asnumpy(amp_noise) if amp_noise is not None else None

        effects = None
        if return_effects:
            effects = WaveletEffects(
                amp_noise=amp_noise_np,
                drift=drift
            )

        return WaveletResult(
            wavelet_train=wavelet_train_np,
            components=components_np,
            amp=amp if return_scaling_factor else None,
            effects=effects
        )

    # -------------------------------
    # Getters
    # -------------------------------

    def get_wavelet_params(self):
        return self.wavelet_params


    def get_config_name(self):
        return self.config_name


    def get_log_params(self):
        return self.log_params
    

    def get_all_params(self):
        all_params = {
            "wavelet_params": self.wavelet_params,
            "config_name": self.config_name,
            "log_params": self.log_params
        }
        return all_params