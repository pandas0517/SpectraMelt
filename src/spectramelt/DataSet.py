from .utils import (
    load_config_from_json,
    get_logger,
    build_flat_paths,
    find_project_root,
    save_to_json
)
import numpy as np

class DataSet:
    def __init__(self,
                 DUT=None,
                 input_sig=None,
                 recovery=None,
                 dataset_config_name="DataSet_Config_1",
                 dataset_params=None,
                 num_input_sigs=5000,
                 tones_per_sig=[1],
                 log_params=None,
                 filenames=None,
                 directories=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------

        """
        if config_file_path is not None:
            dataset_params = load_config_from_json(config_file_path)
        elif dataset_params is None:
            dataset_params = {}
            dataset_params['num_input_sigs'] = num_input_sigs
            dataset_params['tones_per_sig'] = tones_per_sig
            dataset_params['filenames'] = filenames
            dataset_params['directories'] = directories
            dataset_params['config_name'] = dataset_config_name
            dataset_params['log_params'] = log_params

        self.set_dataset_params(dataset_params, DUT, input_sig, recovery)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")

    # -------------------------------
    # Setters
    # -------------------------------
        
    def set_dataset_params(self, dataset_params=None, DUT=None, input_sig=None, recovery=None):
        if dataset_params is None:
            dataset_params = {}
        config_name = dataset_params.get('config_name', "Dataset_Config_1")
        inputset_params = dataset_params.get('inputset_params', None)
        filenames = dataset_params.get('filenames', None)
        directories = dataset_params.get('directories', None)
        log_params = dataset_params.get('log_params', None)
        
        if (filenames is None and
            directories is None ):
            config_name = "Default_Dataset_Config"
            
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
            
        DUT_config_name = "DUT_Config_1"
        input_config_name = "Input_Config_1"
        recovery_config_name = "Recovery_Config_1"
        
        if DUT is not None:
            DUT_config_name = DUT.get_config_name()
        if input_sig is not None:
            input_config_name = input_sig.get_config_name()
        if recovery is not None:
            recovery_config_name = recovery.get_config_name()
            
        self.set_config_name(config_name)
        self.set_DUT_config_name(DUT_config_name)
        self.set_input_config_name(input_config_name)
        self.set_recovery_config_name(recovery_config_name)
        self.set_inputset_params(inputset_params)
        self.set_filenames(filenames)
        self.set_directories(directories)

        
    def set_log_params(self, log_params=None):
        if log_params is None:
            log_params = {
                "enabled": True,
                "log_file": None,
                "level": "INFO",
                "console": True
            }
        self.log_params = log_params
        
        
    def set_config_name(self, config_name):
        self.config_name = config_name
        
        
    def set_inputset_params(self, inputset_params):
        if inputset_params is None:
            inputset_params = {
                "input_freq_range": [1, 1000],
                "num_input_sigs": 5000,
                "tones_per_sig": [1],
                "amp_range": [0.1, 1.0],
                "phase_random": True
            }
        inputset_params["input_freq_range"] = tuple(inputset_params["input_freq_range"])
        self.inputset_params = inputset_params
        
        
    def set_DUT_config_name(self, config_name):
        self.DUT_config_name = config_name
        
        
    def set_input_config_name(self, config_name):
        self.input_config_name = config_name
        
        
    def set_recovery_config_name(self, config_name):
        self.recovery_config_name = config_name
        
            
    def set_directories(self, directories=None):
        if directories is None:
            directories = {}
            directories['dataset_dir'] = "Data_Set"
            directories['paths'] = [
                "inputs",
                "outputs",
                "recovery",
                "ml_models"
            ]
            directories['base'] = {
                "inputs": None,
                "outputs": None,
                "recovery": None,
                "ml_models": None
            }
            directories['tail'] = {
                "inputs": "Inputs",
                "outputs": "Outputs",
                "recovery": "Recovery",
                "ml_models": "ML_Models"
            }

        for base_dir in directories['base']:
            if directories['base'][base_dir] is None:
                directories['base'][base_dir] = find_project_root()
            
        directories['tail']['inputs'] = [self.input_config_name,
                                        directories['tail']['inputs']]
        directories['tail']['outputs'] = [self.input_config_name,
                                        self.DUT_config_name,
                                        directories['tail']['outputs']]
        directories['tail']['recovery'] = [self.input_config_name,
                                            self.DUT_config_name,
                                            self.recovery_config_name,
                                            directories['tail']['recovery']]
        directories['tail']['ml_models'] = [self.input_config_name,
                                    self.DUT_config_name,
                                    self.recovery_config_name,
                                    directories['tail']['ml_models']]
        
        self.directories = build_flat_paths(directories)
        

    def set_filenames(self, filenames=None):
        if filenames is None:
            filenames = {
                "real_time": "real_time.npy",
                "real_freq": "real_freq.npy",
                "wbf_time": "wbf_time.npy",
                "wbf_freq": "wbf_freq.npy",
                "samp_time": "sampled_time.npy",
                "samp_freq": "sampled_freq.npy",
                "input_signal": "signal.npy",
                "input_config": "input_config.json",
                "output_signal": "signal.npy",
                "DUT_config": "DUT_config.json",
                "dictionary": "dictionary.npy",
                "recovered": "recovered.npy",
                "recovery_config": "recovery_config.json",
                "ml_model": "ml_model.keras"
            }

        self.filenames = filenames

    # -------------------------------
    # Core functional methods
    # -------------------------------
        
    def create_input_set(self, input_signal):
        real_time_file = self.directories['inputs'] / self.filenames['real_time']
        real_freq_file = self.directories['inputs'] / self.filenames['real_freq']
        real_time = input_signal.get_analog_time()
        if not (real_time_file).is_file():
            np.save(real_time_file, real_time)
        if not (real_freq_file).is_file():
            np.save(real_freq_file, input_signal.get_analog_frequency())
            
        input_config_file = self.directories['inputs'] / self.filenames['input_config']
        input_sig_params = input_signal.get_input_params()
        if not (input_config_file).is_file():
            save_to_json(input_sig_params, input_config_file)
            
        num_input_sigs = self.num_input_sigs   
        input_signals = np.zeros((num_input_sigs, real_time.size))
        tones_per_sig = self.tones_per_sig
        
        for tones in tones_per_sig: 
            for sig_num in range(num_input_sigs):
             
        wbf_cut_freq = system_params['wbf_cut_freq']
        for noise_level, _ in self.input_set_params["noise_levels"]:
            for phase_shift, _ in self.input_set_params["phase_shifts"]:
                for input_tones, _ in self.input_set_params["input_tones"]:
                    input_file_path = os.path.join(self.input_dir,
                                                   noise_level,
                                                   phase_shift,
                                                   self.input_tones[input_tones]['sigs']) # e.g. 1_2_tone_sigs.npy
                    input_list_path = os.path.join(self.input_dir,
                                                   noise_level,
                                                   phase_shift,
                                                   self.input_tones[input_tones]['list'])
                    input_list = []
                    wave_param_list = []

                    input_list_exists = os.path.isfile(input_list_path)
                    if input_list_exists:
                        with open(input_list_path, 'rb') as file:
                            input_freq_tot_list = pickle.load(file)
                    else:
                        input_freq_tot_list = self.__get_frequency_list(input_tones, wbf_cut_freq)

                    for input_freqs in input_freq_tot_list:
                        if input_list_exists:
                            wave_params = input_freqs[0]
                            noise = input_freqs[1]
                        else:
                            wave_params, noise = self.__update_wave_system(input_freqs,phase_shift,noise_level)
                            wave_param_list.append((wave_params, noise))
                        system_params['system_noise_level'] = noise
                        self.nyfr.set_system_params(system_params=system_params)
                        analog_input, _ = self.nyfr.create_input_signal(wave_params=wave_params)
                        if ( not os.path.isfile(input_file_path) ):
                            input_list.append(self.nyfr.sample_signals(data=analog_input, sample_rate=self.nyfr.get_wb_nyquist_rate()))
                    if wave_param_list:
                        # Save the wave parameters if they were generated
                        with open(input_list_path, 'wb') as file:
                            pickle.dump(wave_param_list, file)
                    if input_list:
                        # Save the input set if it was generated
                        input_set = np.array(input_list)
                        np.save(input_file_path, input_set)
                        

    def create_output_sets(self, nyfr=None, filenames=None, directories=None, input_set_params=None):
        self.__set_init(nyfr, filenames, directories, input_set_params)
        if self.__needs_init(include_set_params=True):
            print("NYFR Test Harness not properly initialized.  Please re-initialize object")
            return
        system_params = self.nyfr.get_system_params()
        LO_params = self.nyfr.get_LO_params()
        for noise_level, _ in self.input_set_params["noise_levels"]:
            for phase_shift, _ in self.input_set_params["phase_shifts"]:
                for input_tones, _ in self.input_set_params["input_tones"]:
                    input_list_path = os.path.join(self.input_dir,
                                                   noise_level,
                                                   phase_shift,
                                                   self.input_tones[input_tones]['list'])
                    for f_mod, f_mod_value in self.input_set_params["f_mods"]:
                        LO_params['phase_freq'] = f_mod_value
                        for f_delta, f_delta_value in self.input_set_params["f_deltas"]:
                            LO_params['phase_delta'] = round(f_delta_value * f_mod_value, 2)
                            self.nyfr.set_LO_params(LO_params=LO_params)
                            output_file_path = os.path.join(self.output_dir,
                                                           noise_level,
                                                           phase_shift,
                                                           f_mod,
                                                           f_delta,
                                                           self.input_tones[input_tones]['sigs'])
                            output_list = []
                            output_file_exists = os.path.isfile(output_file_path)
                            input_list_exists = os.path.isfile(input_list_path)
                            if not output_file_exists and input_list_exists:
                                with open(input_list_path, 'rb') as file:
                                    input_freq_tot_list = pickle.load(file)

                                for input_freqs in input_freq_tot_list:
                                    wave_params = input_freqs[0]
                                    noise = input_freqs[1]

                                    system_params['system_noise_level'] = noise
                                    self.nyfr.set_system_params(system_params=system_params)
                                    analog_input, _ = self.nyfr.create_input_signal(wave_params=wave_params)
                                    output_list.append( self.nyfr.simulate_system(input_signal=analog_input) )

                                output_set = np.array(output_list)
                                np.save(output_file_path, output_set)
                            else:
                                if output_file_exists:
                                    print("Output file already exists: ", output_file_path)
                                if not input_list_exists:
                                    print("Input list file does not exist: ", input_list_path)
                                    

    def batch_recover(self, nyfr=None, filenames=None, directories=None, get_recovery_time=False):
        self.__set_init(nyfr=nyfr, filenames=filenames, directories=directories)
        if self.__needs_init():
            print("NYFR Test Harness not properly initialized.  Please re-initialize object")
            return
        system_params = self.nyfr.get_system_params()
        dictionary_params = self.nyfr.get_dictionary_params()
        recovery_params = self.nyfr.get_recovery_params()

        # mlp_inv_mod = 4 / self.nyfr.get_adc_clock_freq()
        mlp_inv_mod = 1
        recovery_list = []
        for mode in recovery_params['modes']:
            recovery_base_path = self.recovery_dir[dictionary_params['version']][recovery_params['type']]
            if ( recovery_params['type'] == 'MLP1' ):
                recovery_base_path = os.path.join(recovery_base_path,
                                                  mode)
            if ( mode == 'real_imag' ):
                mlp_models_base_path = self.mlp_models_dir[dictionary_params['version']]['real']
                mlp_models_base_path_aux = self.mlp_models_dir[dictionary_params['version']]['imag']
            elif ( mode == 'mag_ang' ):
                mlp_models_base_path = self.mlp_models_dir[dictionary_params['version']]['mag']
                mlp_models_base_path_aux = self.mlp_models_dir[dictionary_params['version']]['ang']
            elif ( mode == 'complex' ):
                mlp_models_base_path = self.mlp_models_dir[dictionary_params['version']]['complex']
                mlp_models_base_path_aux = None
            elif ( mode == 'active_zones' ):
                mlp_models_base_path = self.mlp_models_dir[dictionary_params['version']]['active_zones']
                mlp_models_base_path_aux = None
            for processing_system in system_params['processing_systems']:
                for noise_level, _ in self.input_set_params["noise_levels"]:
                    for phase_shift, _ in self.input_set_params["phase_shifts"]:
                        for input_tones, num_tones in self.input_set_params["input_tones"]:
                            if ( mode == 'real_imag' ):
                                recovery_log_file_path = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['real'][processing_system])
                                recovery_log_file_path_aux = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['imag'][processing_system])
                            elif ( mode == 'mag_ang' ):
                                recovery_log_file_path = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['mag'][processing_system])
                                recovery_log_file_path_aux = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['ang'][processing_system])
                            elif ( mode == 'complex' ):
                                recovery_log_file_path = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode][processing_system])
                                recovery_log_file_path_aux = None
                            elif ( mode == 'active_zones' ):
                                recovery_log_file_path = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode][processing_system])
                                recovery_log_file_path_aux = None
                            for f_mod, _ in self.input_set_params["f_mods"]:
                                for f_delta, _ in self.input_set_params["f_deltas"]:
                                    output_file_path = os.path.join(self.output_dir,
                                                                    noise_level,
                                                                    phase_shift,
                                                                    f_mod,
                                                                    f_delta,
                                                                    self.input_tones[input_tones]['sigs'])
                                    mlp_model_file_path = os.path.join(mlp_models_base_path,
                                                                       noise_level,
                                                                       phase_shift,
                                                                       f_mod,
                                                                       f_delta,
                                                                       self.mlp_models_file['name'])
                                    if self.input_set_params["use_per_signal_model"]:
                                        mlp_model_per_set_file_path = os.path.join(mlp_models_base_path,
                                                                        noise_level,
                                                                        phase_shift,
                                                                        f_mod,
                                                                        f_delta,
                                                                        self.input_tones[input_tones]['sigs'])
                                        mlp_model_per_set_file_path = replace_extension(mlp_model_per_set_file_path, "keras")
                                        replace_file(mlp_model_file_path, mlp_model_per_set_file_path)
                                    mlp_model_aux_file_path = None
                                    if mlp_models_base_path_aux is not None:
                                        mlp_model_aux_file_path = os.path.join(mlp_models_base_path_aux,
                                                                               noise_level,
                                                                               phase_shift,
                                                                               f_mod,
                                                                               f_delta,
                                                                               self.mlp_models_file['name'])
                                    dictionary_file_path = os.path.join(self.dictionary_dir[dictionary_params['version']],
                                                                        f_mod,
                                                                        f_delta,
                                                                        self.dictionary_file['name'])
                                    recovery_file_path = os.path.join(recovery_base_path,
                                                                      noise_level,
                                                                      phase_shift,
                                                                      f_mod,
                                                                      f_delta,
                                                                      self.input_tones[input_tones]['sigs'])
                                    # disabling checking log for processed files flag for now
                                    # found_string_in_file = False
                                    found_string_in_file = True
                                    found_string_in_file_aux = False
                                    if os.path.isfile(recovery_log_file_path):
                                        with open(recovery_log_file_path, "r") as recovery_log:
                                            for line in recovery_log:
                                                if output_file_path in line:
                                                    found_string_in_file = True
                                                    break
                                    if recovery_log_file_path_aux is not None and os.path.isfile(recovery_log_file_path_aux):
                                        with open(recovery_log_file_path_aux, "r") as recovery_log:
                                            for line in recovery_log:
                                                if output_file_path in line:
                                                    found_string_in_file_aux = True
                                                    break                    
                                    if ( found_string_in_file ):
                                        output_set = np.load(output_file_path)
                                        dictionary = np.load(dictionary_file_path)
                                        if ( not os.path.isfile(recovery_file_path )):
                                            if get_recovery_time:
                                                ave_recovery_time = 0
                                                start_time = time.perf_counter()
                                            for idx in range(recovery_params['set_size']):
                                                pass
                                                recovered_signal = self.nyfr.recover_signal(dictionary,
                                                                                            output_set[idx],
                                                                                            file_path=mlp_model_file_path,
                                                                                            aux_file_path=mlp_model_aux_file_path,
                                                                                            mlp_inv_mod=mlp_inv_mod,
                                                                                            mode=mode,
                                                                                            num_tones=num_tones[-1])
                                                recovery_list.append(recovered_signal)
                                            if get_recovery_time:
                                                end_time = time.perf_counter()
                                                ave_recovery_time = ( end_time - start_time ) / recovery_params['set_size']
                                            recovery_set = np.array(recovery_list)
                                            np.save(recovery_file_path, recovery_set)
                                            recovery_list = []
                                        delete_lines_with_string(recovery_log_file_path, output_file_path)
        
    # -------------------------------
    # Getters
    # -------------------------------

    def get_config_name(self):
        return self.config_name
    
    
    def get_DUT_config_name(self):
        return self.DUT_config_name
    
    
    def get_input_config_name(self):
        return self.input_config_name
    
    
    def get_recovery_config_name(self):
        return self.recovery_config_name

    
    def get_directories(self):
        return self.directories

    
    def get_filenames(self):
        return self.filenames


    def get_log_params(self):
        return self.log_params
    
    
    def get_dataset_params(self):
        dataset_params = {
            "config_name": self.config_name,
            "DUT_config_name": self.DUT_config_name,
            "input_config_name": self.input_config_name,
            "reconvery_config_name": self.recovery_config_name,
            "inputset_params": self.inputset_params, 
            "log_params": self.log_params,
            "directories": self.directories,
            "filenames": self.filenames
        }
        return dataset_params