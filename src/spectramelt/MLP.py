import numpy as np
from .utils import load_config_from_json, get_logger

class MLP:
    """
    """

    def __init__(self,
                 input_signal=None,
                 mlp_params=None,
                 recovery_params=None,
                 log_params=None,
                 config_name="MLP_Config_1",
                 config_file_path=None) -> None:
        """
        Parameters
        ----------

        """
        if config_file_path is not None:
            nyfr_params = load_config_from_json(config_file_path)
        elif nyfr_params is None:
            nyfr_params = {}
            nyfr_params['mlp_params'] = lo_params
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
                conditioned_time = self.conditioned_signals.get('time', None)
                self.nyfr_dict = self.create_dictionary(conditioned_time, self.lo_phase_mod_mid)
 
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
        config_name = nyfr_params.get('config_name', "Input_Config_1")
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