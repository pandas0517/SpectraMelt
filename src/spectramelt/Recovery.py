import numpy as np
from importlib import import_module
from .utils import(
    load_config_from_json,
    get_logger,
)
import pickle

VALID_SAVED_FREQ_MODES = {
    "complex", "real", "imag",
    "real_imag", "mag", "ang", "mag_ang"
}

class Recovery:
    VALID_RECOVERY_METHODS = {
        "complex", "real", "imag",
        "real_imag", "mag", "ang", "mag_ang"
    }    

    def __init__(self,
                signal=None,
                dictionary=None,
                recovery_params=None,
                log_params=None,
                config_name=None,
                num_waves=1,
                config_file_path=None) -> None:
        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        else:
            all_params = {}
            all_params["recovery_params"] = recovery_params
            all_params["log_params"] = log_params
            all_params["config_name"] = config_name
            self.set_recovery_params(recovery_params)
            if recovery_params is None:
                config_name = "Default_Recovery_Config"
            self.set_config_name(config_name)
            self.set_log_params(log_params)
        
        self.set_all_params(all_params)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")
                
        self.recovered_coefs = None
        if signal is not None and dictionary is not None:
            self.recovered_coefs = self.recover_signal(signal, dictionary, num_waves)

    # -------------------------------
    # Setters
    # -------------------------------

    def set_all_params(self, all_params=None):
        if all_params is None:
            all_params = {}
            
        recovery_params = all_params.get('recovery_params', None)
        log_params = all_params.get('log_params', None)
        
        config_name = all_params.get('config_name', "Recovery_Config_1")
        if recovery_params is None:
            config_name = "Default_Recovery_Config"
                    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
    
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
                "premultiply": False,
                "recovery_type": "complex",
                "model_file_path": None,
                "sigma": 0.001,
                "dict_mag_adj": 1.0
            }
        self.recovery_params = recovery_params
        self.set_recovery_type(recovery_params.get('recovery_type', None))
        self.set_recovery_method(recovery_params.get('method', None))
        
        
    def set_recovery_type(self, recovery_type):
        if recovery_type is None:
            self.logger.error("Recovery type can not be None")
            raise ValueError("Recovery type can not be None")
        if self.is_valid_saved_freq_mode(recovery_type):
            self.recovery_params["recovery_type"] = recovery_type
        else:
            self.logger.error(f"{recovery_type} Recovery type is not valid")
            raise ValueError(f"{recovery_type} Recovery type is not valid")
        
    
    def set_recovery_method(self, recovery_method):
        if recovery_method is None:
            self.logger.error("Recovery method can not be None")
            raise ValueError("Recovery method can not be None")
        if self.is_valid_recovery_type(recovery_method):
            self.recovery_params["method"] = recovery_method
        else:
            self.logger.error(f"{recovery_method} Recovery method is not valid")
            raise ValueError(f"{recovery_method} Recovery method is not valid")
              
    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def is_valid_saved_freq_mode(self, name) -> bool:
        if isinstance(name, str):
            name = [name]
        return all(n.lower() in VALID_SAVED_FREQ_MODES for n in name)


    def is_valid_recovery_type(self, name) -> bool:
        if isinstance(name, str):
            name = [name]
        return all(n.lower() in self.VALID_RECOVERY_METHODS for n in name)


    def recover_signal(self, signal, dictionary, num_waves=1):
        sigma = self.recovery_params.get('sigma', 0.001)
        dict_mag_adj = self.recovery_params.get('dict_mag_adj', 1.0)
        model_file_path = self.recovery_params.get('model_file_path', None)
        recovery_method = self.recovery_params.get('method', "splg1").lower()
        recovery_type = self.recovery_params.get('recovery_type', "complex").lower()
        premultiply = self.recovery_params.get('premultiply', False)

        recovered_coef = None
        complex_recovery = {"complex", "imag", "ang", "mag_ang", "real_imag"}
        
        if recovery_type in complex_recovery:
            if not np.iscomplexobj(dictionary):
                self.logger.error("Dictionary is not complex")
                raise ValueError("Dictionary is not complex")

        match recovery_method:
            case 'iht':
                IHT = import_module("IHT")
                recovered_coef = IHT.CIHT(dict_mag_adj * dictionary, signal, 2*num_waves, learning_rate=sigma)
            case 'omp':
                OMP = import_module("OMP")
                signal_norm = np.linalg.norm(signal)
                recovered_coef = OMP.OMP(dict_mag_adj * dictionary, signal/signal_norm)[0]
            case 'spgl1':
                spgl1 = import_module("spgl1")
                signal_norm = np.linalg.norm(signal)
                recovered_coef,_,_,_ = spgl1.spgl1(dict_mag_adj * dictionary, signal/signal_norm, sigma=sigma)
            case 'mlp1':
                NYFR_ML_Models = import_module("NYFR_ML_Models")
                if not model_file_path.exists():
                    self.logger.error(f"File not found: {model_file_path}")
                    raise FileNotFoundError(f"File not found: {model_file_path}")
                if premultiply:
                    pseudo = np.linalg.pinv(dict_mag_adj *dictionary)
                    init_guess = np.dot(pseudo,signal)
                else:
                    init_guess = signal
                recovered_coef = NYFR_ML_Models.model_prediction(init_guess, model_file_path)
            case _:
                self.logger.error(f"Recovery method {recovery_method} is not supported")

        self.recovered_coefs = recovered_coef
        return recovered_coef


    def set_recovery_dataframe(self,
                               input_dir,
                               recovery_dir,
                               recovery_file_name,
                               wave_param_file_name,
                               inputset_config_file=None,
                               DUT_config_file=None,
                               recovery_df_file_path=None):
        if input_dir is None:
            self.logger.error("Input directory not specified")
            raise ValueError("Input directory not specified")
        if recovery_dir is None:
            self.logger.error("Recovery directory not specified")
            raise ValueError("Recovery directory not specified")
        if recovery_file_name is None:
            self.logger.error("Recovery file base name not specified")
            raise ValueError("Recovery file base name not specified")
        if wave_param_file_name is None:
            self.logger.error("Input wave parameters file base name not specified")
            raise ValueError("Input wave parameters file base name not specified")
                
        if recovery_df_file_path is None:
            recovery_df_file_path = self.dataframe_params.get('file_path', "recovery_df.pkl")
            
        if not recovery_df_file_path.exists():
            self.logger.error(f"{recovery_df_file_path} does not exist.")
        else:
            recovery_df = pd.read_pickle(recovery_df_file_path)
            
        if inputset_config_file is None:
            if self.inputset_config is None:
                self.logger.error(f"{inputset_config_file} not specified")
                raise ValueError(f"{inputset_config_file} not specified")
            else:
                inputset_config = self.inputset_config
        elif not inputset_config_file.exists():
            self.logger.error(f"{inputset_config_file} does not exist")
            raise ValueError(f"{inputset_config_file} does not exist")
        else:
            inputset_config = load_config_from_json(inputset_config_file)
        
        inputset_config_name = inputset_config.get('config_name')
        num_recovery_sigs = inputset_config.get('num_recovery_sigs')
        
        if DUT_config_file is None:
            if self.DUT_config is None:
                self.logger.error(f"{DUT_config_file} not specified")
                raise ValueError(f"{DUT_config_file} not specified")
            else:
                DUT_config = self.DUT_config
        elif not DUT_config_file.exists():
            self.logger.error(f"{DUT_config_file} does not exist")
            raise ValueError(f"{DUT_config_file} does not exist")
        else:
            DUT_config = load_config_from_json(DUT_config_file)
        
        DUT_config_name = DUT_config.get('config_name')
            
        recovery_dict = {p.name: p for p in recovery_dir.iterdir()
                        if p.is_file() and p.name.endswith(recovery_file_name)
                        and "recovery" in p.name.lower()}

        input_dict = {p.name: p for p in input_dir.iterdir()
                    if p.is_file() and p.name.endswith(recovery_file_name)
                    and "recovery" in p.name.lower()}
        
        input_wave_dict = {p.name: p for p in input_dir.iterdir()
                    if p.is_file() and p.name.endswith(wave_param_file_name)
                    and "recovery" in p.name.lower()}
        
        missing_in_input = recovery_dict.keys() - input_dict.keys()
        missing_in_recovery = input_dict.keys() - recovery_dict.keys()
        missing_in_wave = input_wave_dict.key() - input_dict.keys()

        if missing_in_input:
            self.logger.error("Files missing in input:", missing_in_input)
            raise ValueError("Files missing in input:", missing_in_input)

        if missing_in_recovery:
            self.logger.error("Files missing in recovery:", missing_in_recovery)
            raise ValueError("Files missing in recovery:", missing_in_recovery)
        
        if missing_in_wave:
            self.logger.error("Files missing in input wave parameters:", missing_in_wave)
            raise ValueError("Files missing in input wave parameters:", missing_in_wave)            
        
        matched_recovery_files = [recovery_dict[name] for name in sorted(recovery_dict)]
        matched_input_files = [input_dict[name] for name in sorted(input_dict)]
        matched_wave_file = [input_wave_dict[name] for name in sorted(input_wave_dict)]
        
        for idx, recovery_file in enumerate(matched_recovery_files):
            recovered_signals = np.load(recovery_file)
            
            input_file = matched_input_files[idx]
            input_signals = np.load(input_file)
            
            input_wave_file = matched_wave_file[idx]
            with open(input_wave_file, "rb") as f:
                input_waves = pickle.load(f)
                
            if (recovered_signals.shape[0] != num_recovery_sigs):
                self.logger.error(f"Recovered signal set size {recovered_signals.shape[0]} does not equal expected size {num_recovery_sigs}")
                raise ValueError(f"Recovered signal set size {recovered_signals.shape[0]} does not equal expected size {num_recovery_sigs}")
            if (input_signals.shape[0] != num_recovery_sigs):
                self.logger.error(f"Input signal set size {input_signals.shape[0]} does not equal expected size {num_recovery_sigs}")
                raise ValueError(f"Input signal set size {input_signals.shape[0]} does not equal expected size {num_recovery_sigs}")  
            if (recovered_signals.shape[1] != input_signals.shape[1]):
                self.logger.error(f"Input signal length {input_signals.shape[1]} does not equal Recovered signal length {recovered_signals.shape[1]}")
                raise ValueError(f"Input signal length {input_signals.shape[1]} does not equal Recovered signal length {recovered_signals.shape[1]}")
                                    
        for idx,rec_sig in enumerate(recovery_sig_set):
            meta_data = self.__create_meta_data_dictionary(idx)
            input_sig_param, _ = input_sig_params[idx]
            analog_input, _ = self.nyfr.create_input_signal(wave_params=input_sig_param)
            input_sig = self.nyfr.sample_signals(data=analog_input, sample_rate=self.nyfr.get_wb_nyquist_rate())
            input_sig_xf = fft(input_sig)
            input_sig_tones = np.where(abs(input_sig_xf) > input_tone_thresh)[0]
            input_tone_mag = np.abs(input_sig_xf)
            rec_sig_tones = np.where(abs(rec_sig) > recovery_mag_thresh)[0]
            mask = np.in1d(rec_sig_tones,input_sig_tones)
            recovered_freq = np.where(mask)[0]
            spur_freq = np.where(~mask)[0]
            recovered_tones = rec_sig_tones[recovered_freq]
            spur_tones = rec_sig_tones[spur_freq]
            rec_mag = abs(rec_sig[recovered_tones])
            spur_mag = abs(rec_sig[spur_tones])
            meta_data['num_rec_freq']['value'] = recovered_freq.size
            meta_data['num_spur_freq']['value'] = spur_freq.size
            meta_data['total_input_tones']['value'] = input_sig_tones.size
            meta_data['rec_tone_thresh']['value'] = recovery_mag_thresh
            if ( recovered_freq.size == 0 ):
                meta_data['ave_rec_mag_err']['value'] = -1
                meta_data['ave_rec_mag']['value'] = -1
                meta_data['max_rec_mag']['value'] = -1
                meta_data['min_rec_mag']['value'] = -1
            else:
                meta_data['ave_rec_mag_err']['value'] = abs( np.average(input_tone_mag) - np.average(rec_mag) )
                meta_data['ave_rec_mag']['value'] = np.average(rec_mag)
                meta_data['max_rec_mag']['value'] = np.max(rec_mag)
                meta_data['min_rec_mag']['value'] = np.min(rec_mag)
            if ( spur_freq.size == 0 ):
                meta_data['ave_spur_mag']['value'] = -1
                meta_data['max_spur_mag']['value'] = -1
                meta_data['min_spur_mag']['value'] = -1
            else:
                meta_data['ave_spur_mag']['value'] = np.average(spur_mag)
                meta_data['max_spur_mag']['value'] = np.max(spur_mag)
                meta_data['min_spur_mag']['value'] = np.min(spur_mag)
            pass
            for data in meta_data:
                recovery_df.at[current_recovery_row[0], meta_data[data]['col_name']] = meta_data[data]['value']
            
                          
    def set_recovery_df(self, nyfr=None, filenames=None, directories=None, input_set_params=None):
        self.__set_init(nyfr=nyfr, filenames=filenames, directories=directories, input_set_params=input_set_params)
        if self.__needs_init(include_set_params=True):
            print("NYFR Test Harness not properly initialized.  Please re-initialize object")
            return
        dictionary_params = self.nyfr.get_dictionary_params()
        recovery_params = self.nyfr.get_recovery_params()

        add_columns = False
        
        recovery_df_path = os.path.join(self.df_dir, self.recovery_file['df'])
        if os.path.exists(recovery_df_path):
            recovery_df = pd.read_pickle(recovery_df_path)
            if add_columns:
                self.__add_columns_recovery_df(recovery_df, recovery_params['set_size'], recovery_df_path)
            # recovery_df_file_path_csv = replace_extension(recovery_df_path, "csv")
            # Save the DataFrame to a CSV file
            # recovery_df.to_csv(recovery_df_file_path_csv, index=False)

            for mode in recovery_params['modes']:
                recovery_base_path = self.recovery_dir[dictionary_params['version']][recovery_params['type']]
                if ( recovery_params['type'] == 'MLP1' ):
                    recovery_base_path = os.path.join(recovery_base_path,
                                                    mode)
                for noise_level, _ in self.input_set_params["noise_levels"]:
                    for phase_shift, _ in self.input_set_params["phase_shifts"]:
                        for input_tones, _ in self.input_set_params["input_tones"]:
                            input_list_path = os.path.join(self.input_dir,
                                                           noise_level,
                                                           phase_shift,
                                                           self.input_tones[input_tones]['list'])
                            for f_mod, f_mod_value in self.input_set_params["f_mods"]:
                                for f_delta, f_delta_value in self.input_set_params["f_deltas"]:
                                    recovery_file_path = os.path.join(recovery_base_path,
                                                                      noise_level,
                                                                      phase_shift,
                                                                      f_mod,
                                                                      f_delta,
                                                                      self.input_tones[input_tones]['sigs'])
                                    current_recovery_row = recovery_df.index[(recovery_df['num_tones']==input_tones) &
                                                (recovery_df['noise_level']==noise_level) &
                                                (recovery_df['phase_shift']==phase_shift) &
                                                (recovery_df['f_mod']==f_mod_value) &
                                                (recovery_df['f_delta']==f_delta_value) &
                                                (recovery_df['dictionary_type']==dictionary_params['type']) &
                                                (recovery_df['recovery_method']==recovery_params['type'])]
                                    recovery_df = self.__update_recovery_df(recovery_df,                         
                                                                            recovery_file_path,
                                                                            input_list_path,
                                                                            recovery_params["mag_thresh"],
                                                                            current_recovery_row,
                                                                            self.input_set_params["amp_min"],
                                                                            recovery_params['set_size'])     
                        recovery_df.to_pickle(recovery_df_path)
                        recovery_df_file_path_csv = replace_extension(recovery_df_path, "csv")
                        # Save the DataFrame to a CSV file
                        recovery_df.to_csv(recovery_df_file_path_csv, index=False)
                        

    def __update_recovery_df(self, 
                             recovery_df,
                             recovery_file_path,
                             input_list_path,
                             recovery_mag_thresh,
                             current_recovery_row,
                             input_tone_thresh,
                             recovery_set_size):
        input_sig_params = pd.read_pickle(input_list_path)
        recovery_sig_set = np.load(recovery_file_path)
        system_params = self.nyfr.get_system_params()
        orig_system_noise_level = system_params["system_noise_level"]
        system_params["system_noise_level"] = 0
        self.nyfr.set_system_params(system_params=system_params)

        if (recovery_sig_set.shape[0] != recovery_set_size):
            print("Recovery signal set size does not match expected size")
            return recovery_df
        
        for idx,rec_sig in enumerate(recovery_sig_set):
            meta_data = self.__create_meta_data_dictionary(idx)
            input_sig_param, _ = input_sig_params[idx]
            analog_input, _ = self.nyfr.create_input_signal(wave_params=input_sig_param)
            input_sig = self.nyfr.sample_signals(data=analog_input, sample_rate=self.nyfr.get_wb_nyquist_rate())
            input_sig_xf = fft(input_sig)
            input_sig_tones = np.where(abs(input_sig_xf) > input_tone_thresh)[0]
            input_tone_mag = np.abs(input_sig_xf)
            rec_sig_tones = np.where(abs(rec_sig) > recovery_mag_thresh)[0]
            mask = np.in1d(rec_sig_tones,input_sig_tones)
            recovered_freq = np.where(mask)[0]
            spur_freq = np.where(~mask)[0]
            recovered_tones = rec_sig_tones[recovered_freq]
            spur_tones = rec_sig_tones[spur_freq]
            rec_mag = abs(rec_sig[recovered_tones])
            spur_mag = abs(rec_sig[spur_tones])
            meta_data['num_rec_freq']['value'] = recovered_freq.size
            meta_data['num_spur_freq']['value'] = spur_freq.size
            meta_data['total_input_tones']['value'] = input_sig_tones.size
            meta_data['rec_tone_thresh']['value'] = recovery_mag_thresh
            if ( recovered_freq.size == 0 ):
                meta_data['ave_rec_mag_err']['value'] = -1
                meta_data['ave_rec_mag']['value'] = -1
                meta_data['max_rec_mag']['value'] = -1
                meta_data['min_rec_mag']['value'] = -1
            else:
                meta_data['ave_rec_mag_err']['value'] = abs( np.average(input_tone_mag) - np.average(rec_mag) )
                meta_data['ave_rec_mag']['value'] = np.average(rec_mag)
                meta_data['max_rec_mag']['value'] = np.max(rec_mag)
                meta_data['min_rec_mag']['value'] = np.min(rec_mag)
            if ( spur_freq.size == 0 ):
                meta_data['ave_spur_mag']['value'] = -1
                meta_data['max_spur_mag']['value'] = -1
                meta_data['min_spur_mag']['value'] = -1
            else:
                meta_data['ave_spur_mag']['value'] = np.average(spur_mag)
                meta_data['max_spur_mag']['value'] = np.max(spur_mag)
                meta_data['min_spur_mag']['value'] = np.min(spur_mag)
            pass
            for data in meta_data:
                recovery_df.at[current_recovery_row[0], meta_data[data]['col_name']] = meta_data[data]['value']
        system_params["system_noise_level"] = orig_system_noise_level
        self.nyfr.set_system_params(system_params=system_params)
        return recovery_df

    def __create_meta_data_dictionary(self, idx):
        meta_data = {
            'num_rec_freq': {
                'col_name': "num_rec_freq_" + str(idx),
                'value': 0
            },
            'num_spur_freq': {
                'col_name': "num_spur_freq_" + str(idx),
                'value': 0
            },
            'ave_rec_mag_err': {
                'col_name': "ave_rec_mag_err_" + str(idx),
                'value': 0
            },
            'total_input_tones': {
                'col_name': "total_input_tones_" + str(idx),
                'value': 0
            },
            'rec_tone_thresh': {
                'col_name': "rec_tone_thresh_" + str(idx),
                'value': 0
            },
            'ave_rec_mag': {
                'col_name': "ave_rec_mag_" + str(idx),
                'value': 0
            },
            'max_rec_mag': {
                'col_name': "max_rec_mag_" + str(idx),
                'value': 0
            },
            'min_rec_mag': {
                'col_name': "min_rec_mag_" + str(idx),
                'value': 0
            },
            'ave_spur_mag': {
                'col_name': "ave_spur_mag_" + str(idx),
                'value': 0
            },
            'max_spur_mag': {
                'col_name': "max_spur_mag_" + str(idx),
                'value': 0
            },
            'min_spur_mag': {
                'col_name': "min_spur_mag_" + str(idx),
                'value': 0
            }
        }
        return meta_data

    # -------------------------------
    # Getters
    # -------------------------------

    def get_config_name(self):
        return self.config_name

    
    def get_recovered_coefs(self):
        return self.recovered_coefs

    
    def get_recovery_params(self):
        return self.recovery_params
    
    
    def get_valid_saved_freq_modes(cls):
        return VALID_SAVED_FREQ_MODES


    def get_log_params(self):
        return self.log_params
    
    
    def get_all_params(self):
        all_params ={
            "config_name": self.config_name,
            "recovery_params": self.recovery_params,
            "dataframe_params": self.dataframe_params,
            "log_params": self.log_params
        }
        return all_params