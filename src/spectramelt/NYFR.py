import numpy as np
from .utils import (
    load_config_from_json,
    get_logger,
    filter_valid_names
)
from .results import (
    WBFResult,
    NYFRResult,
    NYFRDictionary
)
import copy
from scipy.linalg import dft
from .LowPassFilter import LowPassFilter
from .PulseGenerator import PulseGenerator
from .LocalOscillator import LocalOscillator
from .ADC import ADC
from .Mixer import Mixer


class NYFR:
    """
    """
    def __init__(self,
                 all_params=None,
                 freq_modes=None,
                 outputset_params=None,
                 lo_params=None,
                 pulse_params=None,
                 mixer_params=None,
                 wbf_params=None,
                 lpf_params=None,
                 adc_params=None,
                 log_params=None,
                 dict_type="real",
                 config_name=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------

        """
        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "freq_modes": freq_modes,
                "outputset_params": outputset_params,
                "lo_params": lo_params,
                "adc_params": adc_params,
                "pulse_params": pulse_params,
                "lpf_params": lpf_params,
                "wbf_params": wbf_params,
                "mixer_params": mixer_params,
                "config_name": config_name,
                "log_params": log_params,
                "dict_type": dict_type
            }

        self.set_all_params(all_params)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")
 
    # -------------------------------
    # Setters
    # -------------------------------
        
    def set_all_params(self, all_params=None):
        if all_params is None:
            all_params = {}

        freq_modes = all_params.get('freq_modes', None)
        outputset_params = all_params.get('outputset_params', None)
        lo_params = all_params.get('lo_params', None)
        adc_params = all_params.get('adc_params', None)
        pulse_params = all_params.get('pulse_params', None)
        lpf_params = all_params.get('lpf_params', None)
        wbf_params = all_params.get('wbf_params', None)
        mixer_params = all_params.get('mixer_params', None)
        dict_type = all_params.get('dict_type', "real")
        log_params = all_params.get('log_params', None)
        
        if ( lo_params is None and 
            adc_params is None and
            pulse_params is None and
            lpf_params is None and
            wbf_params is None and
            mixer_params is None ):
            config_name = "Default_NYFR_Config"
        else:
            config_name = all_params.get('config_name', "NYFR_Config_1")
        
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)

        self.set_freq_modes(freq_modes)
        self.set_outputset_params(outputset_params)
        self.set_lo_params(lo_params)
        self.set_pulse_params(pulse_params)
        self.set_dict_type(dict_type)
        self.set_adc_params(adc_params)
        self.set_config_name(config_name)
        self.set_lpf_params(lpf_params)
        self.set_wbf_params(wbf_params)
        self.set_mixer_params(mixer_params)
        
    
    def set_config_name(self, config_name):
        if config_name is None:
            config_name = "NYFR_Config_1"
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
        
    
    def set_dict_type(self, dict_type):
        # Set Dictionary Type: Real or Complex
        dict_type = dict_type.lower()

        if dict_type not in ("real", "complex"):
            self.logger.error(f"Invalid dict_type '{dict_type}'. Must be 'real' or 'complex'. Defaulting to 'real'")
            dict_type = "real"

        self.dict_type = dict_type
 

    def set_freq_modes(self, freq_modes=None):
        if freq_modes is None:
            freq_modes = {
                "output": [
                    "mag",
                    "ang",
                    "real",
                    "imag"
                ],
                "wideband": [
                    "mag",
                    "ang",
                    "real",
                    "imag",
                    "real_imag",
                    "mag_ang_sincos"
                ]
            }
        
        for freq_mode, freq_mode_list in freq_modes.items():
            valid_modes, removed_modes = filter_valid_names(freq_mode_list)
            freq_modes[freq_mode] = valid_modes
            if removed_modes:
                self.logger.warning(f"Invalid modes removed from {freq_mode} frequency mode list: {removed_modes}")
        self.freq_modes = freq_modes
        
        
    def set_outputset_params(self, outputset_params=None):
        if outputset_params is None:
            outputset_params = {
                "DUT_type": "NYFR",
                "decode_to_time": True,
                "normalize": True,
                "fft_shift": True,
                "normalize_wbf": True,
                "fft_shift_wbf": False,
                "overwrite": False
            }
            
        self.outputset_params = outputset_params
        
    
    def set_lo_params(self, lo_params=None):
        if lo_params is None:
            lo = LocalOscillator()
            lo_params = lo.get_lo_params()     
        self.lo_params = lo_params
        
    
    def set_adc_params(self, adc_params=None):
        if adc_params is None:
            adc = ADC()
            adc_params = adc.get_adc_params()    
        self.adc_params = adc_params

    
    def set_pulse_params(self, pulse_params=None):
        if pulse_params is None:
            pulse = PulseGenerator()
            pulse_params = pulse.get_pulse_params()
        self.pulse_params = pulse_params

    
    def set_lpf_params(self, lpf_params=None):
        if lpf_params is None:
            lpf = LowPassFilter()
            lpf_params = lpf.get_lpf_params()
        self.lpf_params = lpf_params
    
    
    def set_wbf_params(self, wbf_params=None):
        if wbf_params is None:
            wbf = LowPassFilter()
            wbf_params = wbf.get_lpf_params()
            wbf_params["cutoff_freq"] = 100 * wbf_params["cutoff_freq"]
        self.wbf_params = wbf_params
 
    
    def set_mixer_params(self, mixer_params=None):
        if mixer_params is None:
            mixer = Mixer()
            mixer_params = mixer.get_mixer_params()
        self.mixer_params = mixer_params
              
    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def create_dictionary(self, lo_phase_mod_mid, wbf_time):
        """Create a real or complex dictionary matrix efficiently."""

        num_time_points = len(wbf_time)
        time_step = wbf_time[1] - wbf_time[0]
        
        adc_clock_freq = self.adc_params.get('adc_samp_freq', 100)

        # Core band and zone parameters
        K_band = round(num_time_points * adc_clock_freq * time_step)
        if K_band <= 0:
            self.logger.error("Invalid K_band computed for dictionary construction")
            raise ValueError("Invalid K_band computed for dictionary construction")
        Zones = int(num_time_points / K_band)

        R_init = np.eye(K_band, dtype=complex)
        dft_matrix = dft(K_band)

        # Build M_index pattern vectorized
        positive_half_zones = Zones
        M_index = [x for i in range(positive_half_zones) for x in (i, -(i+1))]

        # Choose real or complex dictionary construction
        if self.dict_type == 'complex':
            R, S, PSI = self._create_complex_dict(R_init, M_index, dft_matrix, lo_phase_mod_mid, K_band, Zones)
        else:
            R, S, PSI = self._create_real_dict(R_init, M_index, dft_matrix, lo_phase_mod_mid, K_band, Zones)

        # Final dictionary multiplication
        return NYFRDictionary(
            dictionary=R @ S @ PSI,
            zones=Zones,
            k_bands=K_band
        )


    def _create_complex_dict(self, R_init, M_index,
                             dft_matrix, lo_phase_mod_mid,
                             K_band, Zones):
        """
        Vectorized complex dictionary creation.
        This version has not been tested yet
        """
        lo_mod = lo_phase_mod_mid[:len(M_index)]

        idft_norm = np.conjugate(dft_matrix.T) / (Zones * K_band)
        R = np.tile(R_init, (1, Zones))
        R_row, R_col = R.shape

        # Vectorized exponential diagonal blocks
        exp_factors = np.exp(1j * np.multiply(M_index, lo_mod))
        exp_blocks = np.repeat(exp_factors, K_band)

        S = np.eye(R_col, dtype=complex)
        S *= np.tile(exp_blocks, int(R_col / len(exp_blocks)))[:R_col]

        PSI = np.kron(np.eye(Zones, dtype=complex), idft_norm)
        return R, S, PSI


    def _create_real_dict(self, R_init, M_index,
                          dft_matrix, lo_phase_mod_mid,
                          K_band, Zones):
        """Vectorized real dictionary creation."""
        idft_norm = np.conjugate(dft_matrix.T) / (2 * Zones * K_band)
        R = np.tile(R_init, (1, 2 * Zones))
        R_row, R_col = R.shape

        # Build UL_idft matrix (upper/lower split)
        idft_split = np.hsplit(idft_norm, 2)
        zero_fill = np.zeros_like(idft_split[0])
        UL_idft = np.block([
            [idft_split[0], zero_fill],
            [zero_fill, idft_split[1]]
        ])

        # Precompute modulation terms
        M_index_rev = [-m for m in reversed(M_index)]
        double_M_index = np.array(M_index + M_index_rev)
        lo_mod_concat = np.tile(lo_phase_mod_mid, Zones)
        double_lo_mod = np.tile(lo_mod_concat, 2)

        # Vectorized block-diagonal exponential construction
        S = np.zeros((R_col, R_col), dtype=complex)
        PSI = np.zeros((R_col, R_col // 2), dtype=complex)

        block_indices = np.arange(0, R_col, K_band)
        for idx, i in enumerate(block_indices[:len(double_M_index)]):
            LO_mod = double_lo_mod[i:i + R_row]
            exp_diag = np.exp(1j * double_M_index[idx] * LO_mod)
            S[i:i + R_row, i:i + R_row] = np.diag(exp_diag)

        for i in range(0, R_col, 2 * K_band):
            PSI[i:i + 2 * K_band, i // 2:i // 2 + K_band] = UL_idft

        return R, S, PSI
    
    
    def create_output_signal(self, input_signal, real_time,
                             return_internal=False,
                             return_wbf=False,
                             return_lo=False,
                             return_pulse=False,
                             return_mixed=False,
                             return_lpf=False,
                             return_effects=False):
        
        return_wbf   = return_wbf   or return_internal
        return_lo    = return_lo    or return_internal
        return_pulse = return_pulse or return_internal
        return_mixed = return_mixed or return_internal
        return_lpf   = return_lpf   or return_internal

        wbf = LowPassFilter(lpf_params=self.wbf_params)
        if self.wbf_params is None:
            self.wbf_params = wbf.get_lpf_params()
        wbf_signal = wbf.apply_filter(input_signal, real_time,
                                      return_effects=return_effects)
        
        lo = LocalOscillator(lo_params=self.lo_params)
        if self.lo_params is None:
            self.lo_params = lo.get_lo_params()
        lo_signal = lo.generate_signal(real_time,
                                       return_pre_start=True,
                                       return_phase_mod=True,
                                       return_effects=return_effects)
        
        lo_pre_start = lo_signal.pre_start_lo
        lo_phase_mod = lo_signal.phase_mod
        
        pulse_gen = PulseGenerator(pulse_params=self.pulse_params)
        if self.pulse_params is None:
            self.pulse_params = pulse_gen.get_pulse_params()
        
        pulse_signal = pulse_gen.generate(lo_signal.lo, real_time, lo_pre_start,
                                          return_effects=return_effects)
        
        mixed = Mixer(mixer_params=self.mixer_params)
        if self.mixer_params is None:
            self.mixer_params = mixed.get_mixer_params()
         
        mixed_signal = mixed.mix(wbf_signal.filtered, pulse_signal.pulses,
                                 return_effects=return_effects)
        
        if self.lpf_params is None:
            self.lpf_params = self.wbf_params.copy()
            self.lpf_params['cutoff_freq'] = 100
        lpf = LowPassFilter(lpf_params=self.lpf_params)
        lpf_signal = lpf.apply_filter(mixed_signal.mixed, real_time,
                                      return_effects=return_effects)
        
        adc = ADC(adc_params=self.adc_params)
        if self.adc_params is None:
            self.adc_params = adc.get_adc_params()

        adc_signal = adc.analog_to_digital(lpf_signal.filtered, real_time,
                                            return_conditioned=True,
                                            return_sample_hold=True,
                                            return_effects=return_effects)

        conditioned_time = adc_signal.conditioned.time
        
        wbf_cutoff_freq = self.wbf_params.get('cutoff_freq')
        total_time = abs(conditioned_time[-1] - conditioned_time[0])
        
        # Multiply by 2 to account for Nyquist rate
        points_per_second = round(wbf_cutoff_freq * 2)
        num_time_points = int(round(total_time * points_per_second))

        # real_time = original time vector of wbf_signal
        # Find the indices in real_time that match conditioned_time
        # This works if all values in conditioned_time are guaranteed to be in real_time
        indices = np.nonzero(np.isin(real_time, conditioned_time))[0]

        # Extract the aligned wbf_signal
        wbf_aligned_signal = wbf_signal.filtered[indices]

        # wbf_aligned_signal is now matched to conditioned_time
        # --- Subsample indices ---
        orig_len = len(wbf_aligned_signal)
        indices = np.linspace(0, orig_len - 1, num=num_time_points, dtype=int)

        wbf_signal_sub = wbf_aligned_signal[indices]
        wbf_time = conditioned_time[indices]
        wbf_freq = np.linspace(
            -points_per_second / 2,
            points_per_second / 2,
            num_time_points,
            endpoint=False
        )

        all_wbf_signals = WBFResult(
            wbf_signal=wbf_signal,
            wbf_sub_sig=wbf_signal_sub,
            time=wbf_time,
            freq=wbf_freq
        )

        lo_phase_mod_mid = None
        if isinstance(lo_phase_mod, np.ndarray):
            # Find closest indices of ADC quantizer mid_times in real_time
            sampled_indicies = np.clip(
                np.searchsorted(real_time, adc_signal.quantized.mid_times),
                0,
                len(lo_phase_mod) - 1
            )

            # This really needs to be quantized to add that noise into the dictionary
            # Extract the corresponding lo_phase_mod values
            lo_phase_mod_mid = lo_phase_mod[sampled_indicies]

        return NYFRResult(
            adc_signal=adc_signal,
            wbf_signal=all_wbf_signals if return_wbf else None,
            lo_signal=lo_signal if return_lo else None,
            pulse_signal=pulse_signal if return_pulse else None,
            mixed_signal=mixed_signal if return_mixed else None,
            lpf_signal=lpf_signal if return_lpf else None,
            lo_phase_mod_mid=lo_phase_mod_mid
        )
    
    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_freq_modes(self):
        return copy.deepcopy(self.freq_modes)
    
    
    def get_outputset_params(self):
        return self.outputset_params.copy()
    
    
    def get_lo_params(self):
        return self.lo_params.copy()
    
    
    def get_pulse_params(self):
        return self.pulse_params.copy()
    
    
    def get_adc_params(self):
        return copy.deepcopy(self.adc_params)
     
    
    def get_mixer_params(self):
        return self.mixer_params.copy()
    
    
    def get_lpf_params(self):
        return self.lpf_params.copy()
    
    
    def get_wbf_params(self):
        return self.wbf_params.copy()
    
    
    def get_dict_type(self):
        return self.dict_type
    
    
    def get_config_name(self):
        return self.config_name
    
    
    def get_log_params(self):
        return self.log_params.copy()
    

    def get_all_params(self):
        all_params = {
            "config_name": self.config_name,
            "freq_modes": self.freq_modes,
            "outputset_params": self.outputset_params,
            "lo_params": self.lo_params,
            "mixer_params": self.mixer_params,
            "wbf_params": self.wbf_params,
            "lpf_params": self.lpf_params,
            "pulse_params": self.pulse_params,
            "adc_params": self.adc_params,
            "log_params": self.log_params,
        }
        return copy.deepcopy(all_params)