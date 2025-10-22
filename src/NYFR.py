import numpy as np
from utility import load_settings
from LowPassFilter import LowPassFilter
from PulseGenerator import PulseGenerator
from LocalOscillator import LocalOscillator
from ADC import ADC
from Mixer import Mixer

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
                 nyfr_config_name="NYFR_Config_1",
                 store_internal_sigs=True,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------

        """
        if config_file_path is not None:
            print("Loading NYFR configuration from file: ", config_file_path)
            nyfr_params = load_settings(config_file_path)
        elif nyfr_params is None:
            nyfr_params = {}
            nyfr_params['lo_params'] = lo_params
            nyfr_params['adc_params'] = adc_params
            nyfr_params['pulse_params'] = pulse_params
            nyfr_params['lpf_params'] = lpf_params
            nyfr_params['wbf_params'] = wbf_params
            nyfr_params['mixer_params'] = mixer_params
            nyfr_params['config_name'] = nyfr_config_name
            nyfr_params['store_internal_sigs'] = store_internal_sigs

        self.set_nyfr_params(nyfr_params)
        
        self.conditioned_signal = None
        self.wbf_signal = None
        self.lo_signal = None
        self.pulse_signal = None
        self.mixed_signal = None
        self.lpf_signal = None
        self.sh_signals = None
        self.output_signals = None
        
        if input_signal is not None and real_time is not None:
            self.output_signals = self.create_output_signal(input_signal, real_time)
 
    # -------------------------------
    # Setters
    # -------------------------------
        
    def set_nyfr_params(self, nyfr_config=None):
        if nyfr_config is None:
            nyfr_config = {}
        lo_params = nyfr_config.get('lo_params', None)
        adc_params = nyfr_config.get('adc_params', None)
        pulse_params = nyfr_config.get('pulse_params', None)
        lpf_params = nyfr_config.get('lpf_params', None)
        wbf_params = nyfr_config.get('wbf_params', None)
        mixer_params = nyfr_config.get('mixer_params', None)
        store_internal_sigs = nyfr_config.get('store_internal_sigs', True)
        if ( lo_params is None and 
            adc_params is None and
            pulse_params is None and
            lpf_params is None and
            wbf_params is None and
            mixer_params is None):
            nyfr_config_name = "Default_NYFR_Config"
        else:
            nyfr_config_name = nyfr_config.get('config_name', "NYFR_Config_1")
                    
        self.set_lo_params(lo_params)
        self.set_pulse_params(pulse_params)
        self.set_store_internal_sigs(store_internal_sigs)
        self.set_adc_params(adc_params)
        self.set_nyfr_config_name(nyfr_config_name)
        self.set_lpf_params(lpf_params)
        self.set_wbf_params(wbf_params)
        self.set_mixer_params(mixer_params)

        
    def set_nyfr_config_name(self, nyfr_config_name):
        self.nyfr_config_name = nyfr_config_name
        
    def set_store_internal_sigs(self, store_internal_sigs):
        self.store_internal_sigs = store_internal_sigs
        
    def set_lo_params(self, lo_params):         
        self.lo_params = lo_params
        
    def set_adc_params(self, adc_params):
        if adc_params is not None:
            if self.store_internal_sigs:
                adc_params['store_internal_sigs'] = True           
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
            self.lpf_params = self.wbf_params
            self.lpf_params['cutoff_freq'] = 100
        lpf = LowPassFilter(mixed_signal, real_time, lpf_params=self.lpf_params)
        lpf_signal = lpf.get_signal_out()
        
        adc = ADC(lpf_signal, real_time, adc_params=self.adc_params)
        if self.adc_params is None:
            self.adc_params = adc.get_adc_params()
            
        if self.store_internal_sigs:
            self.wbf_signal = wbf_signal
            self.lo_signal = lo_signal
            self.pulse_signal = pulse_signal
            self.mixed_signal = mixed_signal
            self.lpf_signal = lpf_signal
            self.conditioned_signal = adc.get_conditioned_signal()
            self.sh_signals = adc.get_sh_signals()
            
        quantized_signals = adc.get_quantizer_signals()
        
        return quantized_signals
    
    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_nyfr_params(self):
        nyfr_params = {
            "lo_params": self.lo_params,
            "mixer_params": self.mixer_params,
            "wbf_params": self.wbf_params,
            "lpf_params": self.lpf_params,
            "pulse_params": self.pulse_params,
            "adc_params": self.adc_params,
            "store_interal_sigs": self.store_internal_sigs,
            "nyfr_config_name": self.nyfr_config_name
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
    
    def get_conditioned_signal(self):
        return self.conditioned_signal
    
    def get_sh_signals(self):
        return self.sh_signals
    
    def get_output_signals(self):
        return self.output_signals
    
    def get_nyfr_config_name(self):
        return self.nyfr_config_name
    
    def get_store_internal_sigs(self):
        return self.store_internal_sigs