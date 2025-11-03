from .utils import (
    load_config_from_json,
    get_logger,
    build_flat_paths,
    find_project_root,
    save_to_json
)
import numpy as np
import pickle
import time

class DataSet:
    def __init__(self,
                 input_sig=None,
                 DUT=None,
                 recovery=None,
                 dataset_config_name="DataSet_Config_1",
                 dataset_params=None,
                 inputset_params=None,
                 log_params=None,
                 filenames=None,
                 directory_params=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------

        """
        if config_file_path is not None:
            dataset_params = load_config_from_json(config_file_path)
        elif dataset_params is None:
            dataset_params = {}
            dataset_params['inputset_params'] = inputset_params
            dataset_params['filenames'] = filenames
            dataset_params['directory_params'] = directory_params
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
        directory_params = dataset_params.get('directory_params', None)
        log_params = dataset_params.get('log_params', None)
        
        if (filenames is None and
            directory_params is None ):
            config_name = "Default_Dataset_Config"
            
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
            
        self.set_config_name(config_name)
        self.set_inputset_params(inputset_params)
        self.set_filenames(filenames)
        self.set_directory_params(directory_params)
        self.set_input_sig(input_sig)
        self.set_DUT(DUT)
        self.set_recovery(recovery)
        self.set_ML_models_dir()

        
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
                "num_sigs": 5000,
                "tones_per_sig": [1],
                "wave_precision": None,
                "seed": None
            }
        self.input_rng = np.random.default_rng(inputset_params.get('seed', None))
        self.inputset_params = inputset_params
        

    def set_DUT(self, DUT):
        DUT_config_name = "DUT_Config_1"
        if DUT is not None:
            DUT_config_name = DUT.get_config_name()

        self.directory_params['tail']['outputs'] = [self.input_config_name,
                                        DUT_config_name,
                                        self.directory_params['tail']['outputs']]
        self.directories = build_flat_paths(self.directory_params)
        
        self.DUT_config_name = DUT_config_name
        self.DUT = DUT
        

    def set_input_sig(self, input_sig):
        input_config_name = "Input_Config_1"
        if input_sig is not None:
            input_config_name = input_sig.get_config_name()

        self.directory_params['tail']['inputs'] = [input_config_name,
                                self.directory_params['tail']['inputs']]
        self.directories = build_flat_paths(self.directory_params)

        self.input_config_name = input_config_name
        self.input_sig = input_sig  
        
        
    def set_recovery(self, recovery):
        recovery_config_name = "Recovery_Config_1"
        if recovery is not None:
            recovery_config_name = recovery.get_config_name()

        self.directory_params['tail']['recovery'] = [self.input_config_name,
                                    self.DUT_config_name,
                                    recovery_config_name,
                                    self.directory_params['tail']['recovery']]
        self.directories = build_flat_paths(self.directory_params)

        self.recovery_config_name = recovery_config_name
        self.recovery = recovery  
        

    def set_ML_models_dir(self):
        self.directory_params['tail']['ml_models'] = [self.input_config_name,
                                    self.DUT_config_name,
                                    self.recovery_config_name,
                                    self.directory_params['tail']['ml_models']]
        self.directories = build_flat_paths(self.directory_params)


    def set_directory_params(self, directory_params=None):
        if directory_params is None:
            directory_params = {}
            directory_params['dataset_dir'] = "Data_Set"
            directory_params['paths'] = [
                "inputs",
                "outputs",
                "recovery",
                "ml_models"
            ]
            directory_params['base'] = {
                "inputs": None,
                "outputs": None,
                "recovery": None,
                "ml_models": None
            }
            directory_params['tail'] = {
                "inputs": "Inputs",
                "outputs": "Outputs",
                "recovery": "Recovery",
                "ml_models": "ML_Models"
            }

        for base_dir in directory_params['base']:
            if directory_params['base'][base_dir] is None:
                directory_params['base'][base_dir] = find_project_root()

        self.directory_params = directory_params            
        

    def set_filenames(self, filenames=None):
        if filenames is None:
            filenames = {
                "real_time": "real_time.npy",
                "real_freq": "real_freq.npy",
                "wbf_time": "wbf_time.npy",
                "wbf_freq": "wbf_freq.npy",
                "samp_time": "sampled_time.npy",
                "samp_freq": "sampled_freq.npy",
                "input_signal": "signals.npy",
                "input_wave_params": "wave_params.pkl",
                "inputset_config": "inputset_config.json",
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
        
    def create_input_set(self):
        """
        Generate and save multiple randomized input signals.

        scale (float): resolution multiplier, e.g.
                       1 → 1 Hz steps
                       10 → 0.1 Hz steps
                       100 → 0.01 Hz steps
        """
        self.logger.info("Starting Input Set Creation...")
        # --- Setup and pre-saves ---
        input_signal = self.input_sig
        if input_signal is None:
            self.logger.error("Input Signal Object not set")
            raise ValueError("Input Signal Object Not Set")
        input_dirs = self.directories.get('inputs', "Inputs")
        real_time_filename = self.filenames.get('real_time', "real_time.npy")
        real_freq_filename = self.filenames.get('real_freq', "real_freq.npy")
        inputset_config_filename = self.filenames.get('inputset_config', "inputset_config.json")
        input_wave_params_filename = self.filenames.get('input_wave_params', "wave_params.pkl")
        input_signal_filename = self.filenames.get('input_signal', "signals.npy")

        input_dirs.mkdir(parents=True, exist_ok=True)
        real_time_file = input_dirs / real_time_filename
        real_freq_file = input_dirs / real_freq_filename
        
        real_time = input_signal.get_analog_time()
        real_freq = input_signal.get_analog_frequency()
        if not real_time_file.exists():
            np.save(real_time_file, real_time)
        if not real_freq_file.exists():
            np.save(real_freq_file, real_freq)
        
        # --- Config file ---
        inputset_config_file = input_dirs.parent / inputset_config_filename
        input_signal_params = input_signal.get_input_params()
        input_signal_wave_params = input_signal_params.get('wave_params', None)
        if input_signal_wave_params is None:
            self.logger.error("Input Signal Wave Parameters not set")
            raise ValueError("Input Signal Wave Parameters Not Set")
        
        if not inputset_config_file.exists():
            inputset_config = {
                "config_name": self.config_name,
                "inputset": self.inputset_params,
                "input": input_signal_params
            }
            save_to_json(inputset_config, inputset_config_file)

        # Build discrete frequency grid
        freq_range = input_signal_wave_params.get('freq_range', (100, 1000))
        freq_subset = real_freq[(real_freq >= freq_range[0]) & (real_freq <= freq_range[1])]
        freq_bins = np.copy(freq_subset)

        amp_range = input_signal_wave_params.get('amp_range', (0.1, 1.0))
        phase_range = input_signal_wave_params.get('phase_range', (0, 1))

        num_input_sigs = self.inputset_params.get('num_input_sigs', 5000)   
        tones_per_sig = self.inputset_params.get('tones_per_sig', [1])
        wave_precision = self.inputset_params.get('wave_precision', None)

        # --- Generate all tone sets ---
        for tones in tones_per_sig:
            input_signals = np.zeros((num_input_sigs, real_time.size))
            wave_param_list = [] # reset for this tone set
            start = time.time()
            for input_sig in range(num_input_sigs):
                amps = self.input_rng.uniform(amp_range[0], amp_range[1], tones)
                if wave_precision is not None:
                    amps = np.round(amps, wave_precision)

                # Randomly choose unique bins, then scale back to Hz
                freqs = self.input_rng.choice(freq_bins, size=tones, replace=False)

                # Remove chosen bins from freq_bins (in place)
                freq_bins = freq_bins[~np.isin(freq_bins, freqs)]
                if tones > len(freq_bins):
                    # self.logger.info("Ran out of unique frequency bins for input signals. Resetting...")
                    freq_bins = np.copy(freq_subset)

                if phase_range:
                    t_shift = self.input_rng.uniform(phase_range[0], phase_range[1], tones) / freqs  # seconds
                    phases = 2 * np.pi * freqs * t_shift
                    if wave_precision is not None:
                        phases = np.round(phases, wave_precision)
                else:
                    phases = np.zeros(tones)
                # Save generated wave dictionaries into waves
                wave = [
                    {"amp": float(amps[i]), "freq": float(freqs[i]), "phase": float(phases[i])}
                    for i in range(tones)
                ]
                wave_param_list.append(wave)

                params = input_signal_wave_params
                params["waves"] = wave
                input_signal.set_wave_params(params)
                input_signal.create_input_signal()
                input_signals[input_sig] = input_signal.get_input_signal()

            stop = time.time()
            self.logger.info(f"{num_input_sigs} {tones}-Tone Signal Input Set Creation Time: {stop - start:.6f} seconds")

            # --- Save outputs ---
            inputset_wave_params_path = input_dirs / f"{tones}_tone_{input_wave_params_filename}" 
            with open(inputset_wave_params_path, 'wb') as file:
                pickle.dump(wave_param_list, file)

            inputset_path = input_dirs / f"{tones}_tone_{input_signal_filename}"
            np.save(inputset_path, input_signals)

        self.logger.info("Input Set Creation Complete")                    

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
    

    def get_directory_params(self):
        return self.directory_params
    
    
    def get_dataset_params(self):
        dataset_params = {
            "config_name": self.config_name,
            "DUT_config_name": self.DUT_config_name,
            "input_config_name": self.input_config_name,
            "reconvery_config_name": self.recovery_config_name,
            "inputset_params": self.inputset_params,
            "directory_params": self.directory_params,
            "log_params": self.log_params,
            "directories": self.directories,
            "filenames": self.filenames
        }