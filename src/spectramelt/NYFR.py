import numpy as np
from .utils import load_config_from_json, get_logger
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
                 input_signal=None,
                 real_time=None,
                 nyfr_params=None,
                 lo_params=None,
                 pulse_params=None,
                 mixer_params=None,
                 wbf_params=None,
                 lpf_params=None,
                 adc_params=None,
                 log_params=None,
                 dict_type="real",
                 config_name=None,
                 store_internal_sigs=True,
                 create_dict=True,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------

        """
        if config_file_path is not None:
            nyfr_params = load_config_from_json(config_file_path)
        elif nyfr_params is None:
            nyfr_params = {}
            nyfr_params['lo_params'] = lo_params
            nyfr_params['adc_params'] = adc_params
            nyfr_params['pulse_params'] = pulse_params
            nyfr_params['lpf_params'] = lpf_params
            nyfr_params['wbf_params'] = wbf_params
            nyfr_params['mixer_params'] = mixer_params
            nyfr_params['config_name'] = config_name
            nyfr_params['store_internal_sigs'] = store_internal_sigs
            nyfr_params['dict_type'] = dict_type
            nyfr_params['create_dict'] = create_dict
            nyfr_params['log_params'] = log_params

        self.set_nyfr_params(nyfr_params)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")
        
        self.conditioned_signals = None
        self.wbf_signal = None
        self.lo_signal = None
        self.pulse_signal = None
        self.mixed_signal = None
        self.lpf_signal = None
        self.sh_signals = None
        self.output_signals = None
        self.lo_phase_mod_mid = None
        self.nyfr_dict = None
        self.Zones = None
        self.K_band = None
        self.wbf_time = None
        self.wbf_freq = None
        
        if input_signal is not None and real_time is not None:
            self.output_signals = self.create_output_signal(input_signal, real_time)
            if self.create_dict:
                self.nyfr_dict = self.create_dictionary(self.lo_phase_mod_mid)
 
    # -------------------------------
    # Setters
    # -------------------------------
        
    def set_nyfr_params(self, nyfr_params=None):
        if nyfr_params is None:
            nyfr_params = {}
        store_internal_sigs = nyfr_params.get('store_internal_sigs', True)
        lo_params = nyfr_params.get('lo_params', None)
        adc_params = nyfr_params.get('adc_params', None)
        pulse_params = nyfr_params.get('pulse_params', None)
        lpf_params = nyfr_params.get('lpf_params', None)
        wbf_params = nyfr_params.get('wbf_params', None)
        mixer_params = nyfr_params.get('mixer_params', None)
        config_name = nyfr_params.get('config_name', None)
        dict_type = nyfr_params.get('dict_type', "real")
        create_dict = nyfr_params.get('create_dict', True)
        log_params = nyfr_params.get('log_params', None)
        
        if ( lo_params is None and 
            adc_params is None and
            pulse_params is None and
            lpf_params is None and
            wbf_params is None and
            mixer_params is None ):
            config_name = "Default_Input_Config"
        
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
            
        self.set_store_internal_sigs(store_internal_sigs)
        self.set_lo_params(lo_params)
        self.set_pulse_params(pulse_params)
        self.set_dict_type(dict_type)
        self.set_adc_params(adc_params)
        self.set_config_name(config_name)
        self.set_lpf_params(lpf_params)
        self.set_wbf_params(wbf_params)
        self.set_mixer_params(mixer_params)
        self.set_create_dict(create_dict)
        
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
        
    def set_store_internal_sigs(self, store_internal_sigs):
        self.store_internal_sigs = store_internal_sigs
        
    def set_lo_params(self, lo_params):         
        self.lo_params = lo_params
        
    def set_create_dict(self, create_dict):
        self.create_dict = create_dict
        
    def set_adc_params(self, adc_params):
        if adc_params is not None:
            adc_params['store_conditioned_sigs'] = True
            if self.store_internal_sigs:
                adc_params['store_sh_sigs'] = True           
        self.adc_params = adc_params

    def set_pulse_params(self, pulse_params):
        self.pulse_params = pulse_params

    def set_lpf_params(self, lpf_params):
        self.lpf_params = lpf_params
    
    def set_wbf_params(self, wbf_params):
        self.wbf_params = wbf_params
 
    def set_mixer_params(self, mixer_params):  
        self.mixer_params = mixer_params
              
    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def create_dictionary(self, lo_phase_mod_mid):
        """Create a real or complex dictionary matrix efficiently."""

        num_time_points = len(self.wbf_time)
        time_step = self.wbf_time[1] - self.wbf_time[0]
        
        adc_clock_freq = self.adc_params.get('adc_samp_freq', 100)

        # Core band and zone parameters
        self.K_band = round(num_time_points * adc_clock_freq * time_step)
        self.Zones = int(num_time_points / self.K_band)

        R_init = np.eye(self.K_band, dtype=complex)
        dft_matrix = dft(self.K_band)

        # Build M_index pattern vectorized
        positive_half_zones = self.Zones
        M_index = [x for i in range(positive_half_zones) for x in (i, -(i+1))]

        # Choose real or complex dictionary construction
        if self.dict_type == 'complex':
            R, S, PSI = self._create_complex_dict(R_init, M_index, dft_matrix, lo_phase_mod_mid)
        else:
            R, S, PSI = self._create_real_dict(R_init, M_index, dft_matrix, lo_phase_mod_mid)

        # Final dictionary multiplication
        return R @ S @ PSI


    def _create_complex_dict(self, R_init, M_index, dft_matrix, lo_phase_mod_mid):
        """
        Vectorized complex dictionary creation.
        This version has not been tested yet
        """
        K_band = self.K_band
        Zones = self.Zones
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


    def _create_real_dict(self, R_init, M_index, dft_matrix, lo_phase_mod_mid):
        """Vectorized real dictionary creation."""
        K_band = self.K_band
        Zones = self.Zones

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
    
    def create_output_signal(self, input_signal, real_time):
        wbf = LowPassFilter(input_signal, real_time, lpf_params=self.wbf_params)
        if self.wbf_params is None:
            self.wbf_params = wbf.get_lpf_params()
        wbf_signal = wbf.get_signal_out()
        
        lo = LocalOscillator(real_time, lo_params=self.lo_params)
        if self.lo_params is None:
            self.lo_params = lo.get_lo_params()
        lo_signal = lo.get_lo_signal()
        lo_pre_start = lo.get_pre_start_lo()
        lo_phase_mod = lo.get_phase_mod()
        
        pulse_gen = PulseGenerator(lo_signal, real_time, lo_pre_start,
                                    pulse_params=self.pulse_params)
        if self.pulse_params is None:
            self.pulse_params = pulse_gen.get_pulse_params()
        pulse_signal = pulse_gen.get_pulse_signal()
        
        mixed = Mixer(wbf_signal, pulse_signal, mixer_params=self.mixer_params)
        if self.mixer_params is None:
            self.mixer_params = mixed.get_mixer_params()
        mixed_signal = mixed.get_mixed_signal()
        
        if self.lpf_params is None:
            self.lpf_params = self.wbf_params.copy()
            self.lpf_params['cutoff_freq'] = 100
        lpf = LowPassFilter(mixed_signal, real_time, lpf_params=self.lpf_params)
        lpf_signal = lpf.get_signal_out()
        
        adc = ADC(lpf_signal, real_time, adc_params=self.adc_params)
        if self.adc_params is None:
            self.adc_params = adc.get_adc_params()
            
        quantized_signals = adc.get_quantizer_signals()
        self.conditioned_signals = adc.get_conditioned_signals()
        conditioned_time = self.conditioned_signals.get('time', None)
        
        wbf_cutoff_freq = self.wbf_params.get('cutoff_freq')
        total_time = abs(conditioned_time[-1] - conditioned_time[0])
        
        # Multiply by 2 to account for Nyquist rate
        points_per_second = round(wbf_cutoff_freq * 2)
        num_time_points = int(round(total_time * points_per_second))        
        # real_time = original time vector of wbf_signal
        # Find the indices in real_time that match conditioned_time
        # This works if all values in conditioned_time are guaranteed to be in real_time
        indices = np.nonzero(np.in1d(real_time, conditioned_time))[0]

        # Extract the aligned wbf_signal
        wbf_aligned_signal = wbf_signal[indices]

        # wbf_aligned_signal is now matched to conditioned_time
        # --- Subsample indices ---
        orig_len = len(wbf_aligned_signal)
        indices = np.linspace(0, orig_len - 1, num=num_time_points, dtype=int)

        self.wbf_signal_sub = wbf_aligned_signal[indices]
        self.wbf_time = conditioned_time[indices]

        self.wbf_freq = np.linspace(
            -points_per_second / 2,
            points_per_second / 2,
            num_time_points,
            endpoint=False
        )
        
        # Find closest indices of ADC quantizer mid_times in real_time
        sampled_indicies = np.searchsorted(real_time, quantized_signals.get('mid_times'))
        
        if isinstance(lo_phase_mod, np.ndarray):
            # Extract the corresponding lo_phase_mod values
            self.lo_phase_mod_mid = lo_phase_mod[sampled_indicies]
         
        if self.store_internal_sigs:
            self.logger.debug("Storing Internal Signals")
            self.wbf_signal = wbf_signal
            self.lo_signal = lo_signal
            self.pulse_signal = pulse_signal
            self.mixed_signal = mixed_signal
            self.lpf_signal = lpf_signal
            self.sh_signals = adc.get_sh_signals()
            
        return quantized_signals
    
    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_all_params(self):
        nyfr_params = {
            "config_name": self.config_name,
            "lo_params": self.lo_params,
            "mixer_params": self.mixer_params,
            "wbf_params": self.wbf_params,
            "lpf_params": self.lpf_params,
            "pulse_params": self.pulse_params,
            "adc_params": self.adc_params,
            "log_params": self.log_params,
            "store_interal_sigs": self.store_internal_sigs
        }
        return nyfr_params
    
    def get_lo_params(self):
        return self.lo_params
    
    def get_pulse_params(self):
        return self.pulse_params
    
    def get_adc_params(self):
        return self.adc_params
     
    def get_mixer_params(self):
        return self.mixer_params
    
    def get_lpf_params(self):
        return self.lpf_params
    
    def get_wbf_params(self):
        return self.wbf_params
    
    def get_wbf_signal(self):
        return self.wbf_signal
    
    def get_lo_signal(self):
        return self.lo_signal
    
    def get_pulse_signal(self):
        return self.pulse_signal
    
    def get_mixed_signal(self):
        return self.mixed_signal
    
    def get_lpf_signal(self):
        return self.lpf_signal
    
    def get_conditioned_signals(self):
        return self.conditioned_signals
    
    def get_sh_signals(self):
        return self.sh_signals
    
    def get_lo_phase_mod_mid(self):
        return self.lo_phase_mod_mid
    
    def get_wbf_signal_sub(self):
        return self.wbf_signal_sub
    
    def get_wbf_time(self):
        return self.wbf_time
    
    def get_wbf_freq(self):
        return self.wbf_freq
    
    def get_nyfr_dict(self):
        return self.nyfr_dict
    
    def get_nyfr_zones(self):
        return self.Zones
    
    def get_nyfr_k_bands(self):
        return self.K_band
    
    def get_output_signals(self):
        return self.output_signals
    
    def get_dict_type(self):
        return self.dict_type
    
    def get_config_name(self):
        return self.config_name
    
    def get_store_internal_sigs(self):
        return self.store_internal_sigs
    
    def get_log_params(self):
        return self.log_params