from utility import load_settings
import warnings

class DataSet:
    def __init__(self,
                 DUT=None,
                 input_sig=None,
                 dataset_config_name="DataSet_Config_1",
                 dataset_params=None,
                 filenames=None,
                 directories=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------

        """
        if config_file_path is not None:
            print("Loading NYFR configuration from file: ", config_file_path)
            dataset_params = load_settings(config_file_path)
        elif dataset_params is None:
            dataset_params = {}
            dataset_params['filenames'] = filenames
            dataset_params['directories'] = directories
            dataset_params['config_name'] = dataset_config_name

        self.set_dataset_params(dataset_params)

    # -------------------------------
    # Setters
    # -------------------------------
        
    def set_dataset_params(self, dataset_params=None):
        if dataset_params is None:
            warnings.warn("No dataset parameters given")
            dataset_params = {}
        store_internal_sigs = nyfr_params.get('store_internal_sigs', True)
        lo_params = nyfr_params.get('lo_params', None)
        adc_params = nyfr_params.get('adc_params', None)
        pulse_params = nyfr_params.get('pulse_params', None)
        lpf_params = nyfr_params.get('lpf_params', None)
        wbf_params = nyfr_params.get('wbf_params', None)
        mixer_params = nyfr_params.get('mixer_params', None)
        nyfr_config_name = nyfr_params.get('config_name', "Input_Config_1")
        dict_type = nyfr_params.get('dict_type', "real")
        create_dict = nyfr_params.get('create_dict', True)
        
        if ( lo_params is None and 
            adc_params is None and
            pulse_params is None and
            lpf_params is None and
            wbf_params is None and
            mixer_params is None ):
            nyfr_config_name = "Default_Input_Config"
            
        self.set_store_internal_sigs(store_internal_sigs)
        self.set_lo_params(lo_params)
        self.set_pulse_params(pulse_params)
        self.set_dict_type(dict_type)
        self.set_adc_params(adc_params)
        self.set_nyfr_config_name(nyfr_config_name)
        self.set_lpf_params(lpf_params)
        self.set_wbf_params(wbf_params)
        self.set_mixer_params(mixer_params)
        self.set_create_dict(create_dict)
        
    def set_nyfr_config_name(self, nyfr_config_name):
        self.nyfr_config_name = nyfr_config_name