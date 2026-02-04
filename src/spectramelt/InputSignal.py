from .utils import (
    load_config_from_json,
    get_logger,
    filter_valid_names
)
from .results import (
    InputSignalEffects,
    InputSignalResult
)
import copy
import numpy as np


class InputSignal:
    def __init__(self,
                 all_params=None,
                 freq_modes=None,
                 inputset_params=None,
                 env_params=None,
                 wave_params=None,
                 log_params=None,
                 config_name=None,
                 config_file_path=None) -> None:

        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "freq_modes": freq_modes,
                "inputset_params": inputset_params,
                "env_params": env_params,
                "wave_params": wave_params,
                "config_name": config_name,
                "log_params": log_params
            }

        self.set_all_params(all_params)

        if config_file_path is not None and self.logger is not None:
            self.logger.info(
                f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}"
            )

    # -------------------------------
    # Setters
    # -------------------------------

    def set_all_params(self, all_params=None):
        if all_params is None:
            all_params = {}

        freq_modes = all_params.get('freq_modes', None)
        inputset_params = all_params.get('inputset_params', None)
        env_params = all_params.get('env_params', None)
        wave_params = all_params.get('wave_params', None)
        log_params = all_params.get('log_params', None)
        config_name = all_params.get('config_name', None)

        if env_params is None and wave_params is None:
            config_name = "Default_Input_Config"
        else:
            config_name = all_params.get('config_name', "Input_Config_1")

        self.set_log_params(log_params)

        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)

        self.set_freq_modes(freq_modes)
        self.set_inputset_params(inputset_params)
        self.set_env_params(env_params)
        self.set_wave_params(wave_params)
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

    def set_freq_modes(self, freq_modes=None):
        if freq_modes is None:
            freq_modes = ["mag", "ang", "real", "imag"]

        valid_modes, removed_modes = filter_valid_names(freq_modes)
        if removed_modes and self.logger is not None:
            self.logger.warning(
                f"Invalid modes removed from frequency mode list: {removed_modes}"
            )
        self.freq_modes = valid_modes

    def set_inputset_params(self, inputset_params=None):
        if inputset_params is None:
            inputset_params = {
                "num_sigs": 4000,
                "num_recovery_sigs": 100,
                "wave_precision": 3,
                "tones_per_sig": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "normalize": True,
                "fft_shift": True,
                "overwrite": True
            }
        self.inputset_params = inputset_params

    def set_env_params(self, env_params=None):
        if env_params is None:
            env_params = {
                "noise_level": 0.0,
                "attenuation": 1.0,
                "doppler": 0.0,
                "delay": 0.0,
                "num_echoes": 0,
                "max_delay": 0.0,
                "max_doppler": 0.0,
                "seed": None
            }
        self.rng = np.random.default_rng(env_params.get('seed', None))
        self.env_params = env_params

    def set_wave_params(self, wave_params=None):
        if wave_params is None:
            wave_params = {
                "num_waves": 1,
                "freq_range": [100, 1000],
                "amp_range": [0.1, 1.0],
                "phase_range": [0, 1],
                "v_ref_range": [0, 1],
                "allow_clipping": False,
                "waves": [
                    {"amp": 1, "freq": 50, "phase": 0},
                    {"amp": 1, "freq": 500, "phase": 0},
                    {"amp": 1, "freq": 1500, "phase": 0}
                ]
            }

        wave_params['phase_range'] = tuple(wave_params['phase_range'])
        wave_params["freq_range"] = tuple(wave_params["freq_range"])
        wave_params["amp_range"] = tuple(wave_params["amp_range"])
        wave_params["v_ref_range"] = tuple(wave_params["v_ref_range"])
        self.wave_params = wave_params

    # -------------------------------
    # Core functional methods
    # -------------------------------

    def create_input_signal(self, real_time, return_effects=False) -> InputSignalResult:
        """
        Generate composite signal with environmental effects and random wave generation.
        """
        if real_time is None:
            if self.logger is not None:
                self.logger.error("Must provide real time signal")
            raise ValueError("Must provide real time signal")

        # Use a local waves list to avoid mutating config
        waves = self.wave_params.get('waves', [])
        num_waves = self.wave_params.get('num_waves', 1)
        freq_range = self.wave_params.get('freq_range', (100, 1000))
        amp_range = self.wave_params.get('amp_range', (0.1, 1.0))
        phase_range = self.wave_params.get('phase_range', (0, 1))
        allow_clipping = self.wave_params.get('allow_clipping', False)
        v_ref_range = tuple(self.wave_params.get('v_ref_range', (0, 1)))

        # === Wave parameter setup ===
        if not waves:
            amps = self.rng.uniform(amp_range[0], amp_range[1], num_waves)
            freqs = self.rng.uniform(freq_range[0], freq_range[1], num_waves)

            if phase_range:
                t_shift = self.rng.uniform(phase_range[0] / freqs, phase_range[1] / freqs)
                phases = 2 * np.pi * freqs * t_shift
            else:
                phases = np.zeros(num_waves)

            waves = [
                {"amp": float(amps[i]), "freq": float(freqs[i]), "phase": float(phases[i])}
                for i in range(num_waves)
            ]

        input_signal, env_effects = self._generate_signal(waves, real_time)

        # --- Adjust to midpoint ---
        midpoint = (v_ref_range[1] + v_ref_range[0]) / 2
        input_signal = input_signal + midpoint

        # --- Scale signal ---
        if not allow_clipping:
            max_abs_val = max(abs(input_signal.min() - midpoint), abs(input_signal.max() - midpoint))
            max_allowed_amp = (v_ref_range[1] + v_ref_range[0]) / 2
            if max_abs_val > 0:
                scale = max_allowed_amp / max_abs_val
                input_signal = (input_signal - midpoint) * scale + midpoint

        return InputSignalResult(
            input_signal=input_signal,
            effects=env_effects if return_effects else None
        )

    def _generate_signal(self, waves, real_time):
        amps = np.array([wave['amp'] for wave in waves])
        freqs = np.array([wave['freq'] for wave in waves])
        phases = np.array([wave['phase'] for wave in waves])

        # --- Set Environment Nonidealities
        doppler = self.env_params.get('doppler', 0.0)
        delay = self.env_params.get('delay', 0.0)
        attenuation = self.env_params.get('attenuation', 1.0)
        noise_level = self.env_params.get('noise_level', 0.0)
        num_echoes = self.env_params.get('num_echoes', 0)
        max_delay = self.env_params.get('max_delay', 0.01)
        max_doppler = self.env_params.get('max_doppler', 0.002)
        phase_inversion_prob = self.env_params.get('phase_inversion_prob', 0.5)

        # === Apply system-level Doppler and delay ===
        freqs = freqs * (1 + doppler)
        real_time = real_time + delay

        # === Generate base composite signal ===
        signals = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * real_time + phases[:, None])
        signal = np.sum(signals, axis=0)

        # === Environmental attenuation ===
        signal *= attenuation

        # === Add Gaussian noise ===
        noise = None
        if noise_level > 0:
            noise = self.rng.normal(0, noise_level, signal.shape)
            signal += noise

        # ===== Multipath reflections =====
        delays = []
        echo_atts = []
        local_dopplers = []
        phase_inversions = []

        if num_echoes > 0:
            dt = real_time[1] - real_time[0]

            for _ in range(num_echoes):
                delay_i = self.rng.uniform(0, max_delay)
                shift = int(delay_i / dt)

                echo_att_i = self.rng.uniform(0.2, 0.8)

                local_doppler_i = 1 + self.rng.uniform(-max_doppler, max_doppler)
                real_time_echo = real_time * local_doppler_i
                echo = np.interp(real_time, real_time_echo, signal, left=0, right=0) * echo_att_i

                phase_inversion_i = False
                if self.rng.random() < phase_inversion_prob:
                    phase_inversion_i = True
                    echo = -echo

                echo = np.roll(echo, shift)
                echo[:shift] = 0

                signal += echo

                delays.append(delay_i)
                echo_atts.append(echo_att_i)
                local_dopplers.append(local_doppler_i)
                phase_inversions.append(phase_inversion_i)

        effects = InputSignalEffects(
            noise=noise,
            delay=tuple(delays) if delays else None,
            echo_att=tuple(echo_atts) if echo_atts else None,
            local_doppler=tuple(local_dopplers) if local_dopplers else None,
            phase_inversion=tuple(phase_inversions) if phase_inversions else None
        )
        return signal, effects

    # -------------------------------
    # Getters
    # -------------------------------

    def get_config_name(self):
        return self.config_name

    def get_freq_modes(self):
        return self.freq_modes.copy()

    def get_inputset_params(self):
        return copy.deepcopy(self.inputset_params)

    def get_env_params(self):
        return self.env_params.copy()

    def get_wave_params(self):
        return copy.deepcopy(self.wave_params)

    def get_log_params(self):
        return self.log_params.copy()

    def get_all_params(self):
        all_params = {
            "config_name": self.config_name,
            "freq_modes": self.freq_modes,
            "inputset_params": self.inputset_params,
            "env_params": self.env_params,
            "wave_params": self.wave_params,
            "log_params": self.log_params
        }
        return copy.deepcopy(all_params)