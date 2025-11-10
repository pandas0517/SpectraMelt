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
import copy
from importlib import import_module
from scipy.fft import fft, fftshift

class DataSet:
    def __init__(self,
                 input_config_name=None,
                 DUT_config_name=None,
                 recovery_config_name=None,
                 ML_config_name=None,
                 dataset_config_name="DataSet_Config_1",
                 dataset_params=None,
                 inputset_params=None,
                 outputset_params=None,
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
            dataset_params['outputset_params'] = outputset_params
            dataset_params['filenames'] = filenames
            dataset_params['directory_params'] = directory_params
            dataset_params['config_name'] = dataset_config_name
            dataset_params['log_params'] = log_params
            
        dataset_params['input_config_name'] = input_config_name
        dataset_params['DUT_config_name'] = DUT_config_name
        dataset_params['recovery_config_name'] = recovery_config_name
        dataset_params['ML_config_name'] = ML_config_name

        self.set_dataset_params(dataset_params)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")

    # -------------------------------
    # Setters
    # -------------------------------
        
    def set_dataset_params(self, dataset_params=None):
        if dataset_params is None:
            dataset_params = {}
        config_name = dataset_params.get('config_name', "Dataset_Config_1")
        input_config_name = dataset_params.get('input_config_name', None)
        DUT_config_name = dataset_params.get('DUT_config_name', None)
        recovery_config_name = dataset_params.get('recovery_config_name', None)
        ML_config_name = dataset_params.get('ML_config_name', None)
        inputset_params = dataset_params.get('inputset_params', None)
        outputset_params = dataset_params.get('outputset_params', None)
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
        self.set_outputset_params(outputset_params)
        self.set_filenames(filenames)
        self.set_directory_params(directory_params)
        self.set_input_config_name(input_config_name)
        self.set_DUT_config_name(DUT_config_name)
        self.set_recovery_config_name(recovery_config_name)
        self.set_ML_config_name(ML_config_name)

        
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
        

    def set_outputset_params(self, outputset_params):
        if outputset_params is None:
            outputset_params = {
                "DUT_type": "NYFR",
                "valid_types": [
                    "NYFR"
                ],
                "scale_dict": 1.0
            }
        
        valid_types = outputset_params.get('valid_types', ["nyfr"])
        outputset_params['valid_types'] = [w.lower() for w in valid_types]
                    
        self.outputset_params = outputset_params


    def set_input_config_name(self, input_config_name):
        if input_config_name is None:
            input_config_name = "Input_Config_1"

        self.internal_directory_params['tail']['inputs'] = [input_config_name,
                                self.directory_params['tail']['inputs']]
        
        self.directories = build_flat_paths(self.internal_directory_params)
        
        self.input_config_name = input_config_name
        

    def set_DUT_config_name(self, DUT_config_name):
        if DUT_config_name is None:
            DUT_config_name = "DUT_Config_1"

        self.internal_directory_params['tail']['premultiply'] = [self.input_config_name,
                                    DUT_config_name,
                                    self.directory_params['tail']['outputs'],
                                    self.directory_params['tail']['premultiply']]
        self.internal_directory_params['tail']['outputs'] = [self.input_config_name,
                                    DUT_config_name,
                                    self.directory_params['tail']['outputs']]

        self.directories = build_flat_paths(self.internal_directory_params)
        
        self.DUT_config_name = DUT_config_name
     

    def set_recovery_config_name(self, recovery_config_name):
        if recovery_config_name is None:
            recovery_config_name = "Recovery_Config_1"

        self.internal_directory_params['tail']['recovery'] = [self.input_config_name,
                                    self.DUT_config_name,
                                    recovery_config_name,
                                    self.directory_params['tail']['recovery']]
        
        self.directories = build_flat_paths(self.internal_directory_params)

        self.recovery_config_name = recovery_config_name 
        

    def set_ML_config_name(self, ML_config_name):
        if ML_config_name is None:
            ML_config_name = "ML_Config_1"

        self.internal_directory_params['tail']['ml_models'] = [self.input_config_name,
                                    self.DUT_config_name,
                                    ML_config_name,
                                    self.directory_params['tail']['ml_models']]
        
        self.directories = build_flat_paths(self.internal_directory_params)
        
        self.ML_config_name = ML_config_name


    def set_directory_params(self, directory_params=None):
        if directory_params is None:
            directory_params = {}
            directory_params['dataset_dir'] = "Data_Set"
            directory_params['paths'] = [
                "inputs",
                "outputs",
                "premultiply",
                "recovery",
                "ml_models"
            ]
            directory_params['base'] = {
                "inputs": None,
                "outputs": None,
                "premultiply": None,
                "recovery": None,
                "ml_models": None
            }
            directory_params['tail'] = {
                "inputs": "Inputs",
                "outputs": "Outputs",
                "premultiply": "Premultiply",
                "recovery": "Recovery",
                "ml_models": "ML_Models"
            }

        for base_dir in directory_params['base']:
            if directory_params['base'][base_dir] is None:
                directory_params['base'][base_dir] = find_project_root()

        self.internal_directory_params = copy.deepcopy(directory_params)
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
                "input_time_signal": "time_signals.npy",
                "input_freq_signal": "freq_signals.npy",
                "input_wave_params": "wave_params.pkl",
                "inputset_config": "inputset_config.json",
                "output_signal": "time_signals.npy",
                "premultiply": "premult_signals.npy",
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
        """
        Generate and save multiple randomized input signals.

        scale (float): resolution multiplier, e.g.
                       1 → 1 Hz steps
                       10 → 0.1 Hz steps
                       100 → 0.01 Hz steps
        """
        self.logger.info("Starting Input Set Creation...")
        # --- Setup and pre-saves ---
        if input_signal is None:
            self.logger.error("Input Signal Object not set")
            raise ValueError("Input Signal Object Not Set")
        else:
            self.set_input_config_name(input_signal.get_config_name())
            
        input_dirs = self.directories.get('inputs', "Inputs")
        real_time_filename = self.filenames.get('real_time', "real_time.npy")
        real_freq_filename = self.filenames.get('real_freq', "real_freq.npy")
        inputset_config_filename = self.filenames.get('inputset_config', "inputset_config.json")
        input_wave_params_filename = self.filenames.get('input_wave_params', "wave_params.pkl")
        input_time_signal_filename = self.filenames.get('input_time_signal', "time_signals.npy")
        input_freq_signal_filename = self.filenames.get('input_freq_signal', "freq_signals.npy")

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
        input_signal_params = input_signal.get_all_params()
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
            input_signals_time = np.zeros((num_input_sigs, real_time.size))
            input_signals_freq = np.zeros((num_input_sigs, real_time.size))
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
                input_signal_time = input_signal.get_input_signal()
                input_signal_freq = (fft(input_signal_time)) / len(real_freq)
                input_signals_time[input_sig] = input_signal_time
                input_signals_freq[input_sig] = input_signal_freq

            stop = time.time()
            self.logger.info(f"{num_input_sigs} {tones}-Tone Signal Input Set Creation Time: {stop - start:.6f} seconds")

            # --- Save outputs ---
            inputset_wave_params_path = input_dirs / f"{tones}_tone_{input_wave_params_filename}" 
            with open(inputset_wave_params_path, 'wb') as file:
                pickle.dump(wave_param_list, file)
                
            inputset_time_path = input_dirs / f"{tones}_tone_{input_time_signal_filename}"
            inputset_freq_path = input_dirs / f"{tones}_tone_{input_freq_signal_filename}"
            
            np.save(inputset_time_path, input_signals_time)
            self.logger.info(f"{tones}-Tone Time Input Set saved to file {inputset_time_path}")
            
            np.save(inputset_freq_path, input_signals_freq)
            self.logger.info(f"{tones}-Tone Frequency Input Set saved to file {inputset_freq_path}") 

        self.logger.info("All Input Sets Created and Saved")


    def create_output_set(self, DUT, input_signal=None):
        self.logger.info(f"Starting Output Set Creation...")
        
        # --- Setup and pre-saves ---
        if input_signal is not None:
            self.set_input_config_name(input_signal.get_config_name())
            
        if DUT is None:
            self.logger.error("DUT Object not set")
            raise ValueError("DUT Object Not Set")
        else:
            DUT_type = type(DUT).__name__.lower()
            if DUT_type in self.outputset_params['valid_types']:
                self.outputset_params['DUT_type'] = DUT_type
            else:
                self.logger.error("DUT not a valid type")
                raise ValueError("DUT not a valid type")
            self.set_DUT_config_name(DUT.get_config_name())
        
        input_dir = self.directories.get('inputs', "Inputs")
        input_time_signal_filename = self.filenames.get('input_time_signal', "time_signals.npy")
        real_time_filename = self.filenames.get('real_time', "real_time.npy")
        real_time_file = input_dir / real_time_filename
        if real_time_file.exists():
            real_time = np.load(real_time_file)
        elif input_signal is not None:
            real_time = input_signal.get_analog_time()
        else:
            self.logger.error("No time file found and Input Signal Object not set")
            raise ValueError("No time file found and Input Signal Object Not Set")
        
        output_dirs = self.directories.get('outputs', "Outputs")
        
        DUT_config_filename = self.filenames.get('DUT_config', "DUT_config.json")
        output_signal_filename = self.filenames.get("output_signal", "signals.npy")
        dictionary_filename = self.filenames.get('dictionary',"dictionary.npy")
        
        samp_time_filename = self.filenames.get('samp_time', "sampled_time.npy")
        samp_freq_filename = self.filenames.get('samp_freq', "sampled_freq.npy")
        
        wbf_time_filename = self.filenames.get('wbf_time', "wbf_time.npy")
        wbf_freq_filename = self.filenames.get('wbf_freq', "wbf_freq.npy")
        
        output_dirs.mkdir(parents=True, exist_ok=True)
        output_signal_file = output_dirs / f"{key_part}{output_signal_filename}"
        
        dictionary_file = output_dirs / dictionary_filename
        
        samp_time_file = output_dirs / samp_time_filename
        samp_freq_file = output_dirs / samp_freq_filename 
        
        wbf_time_file = output_dirs / wbf_time_filename
        wbf_freq_file = output_dirs / wbf_freq_filename
        
        # --- Config file ---
        DUT_config_file = output_dirs.parent / DUT_config_filename
        DUT_params = DUT.get_all_params()
        outputset_params = self.outputset_params
        if not DUT_config_file.exists():
            DUT_config = {
                "config_name": self.config_name,
                "output": DUT_params,
                "outputset": outputset_params
            }
            save_to_json(DUT_config, DUT_config_file)
            
        DUT_type = outputset_params.get('DUT_type', "nyfr")        
        
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
          
                input_signals = np.load(file_path)
                
                # Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(input_time_signal_filename)[0]
                
                dictionary = None
                output_signal_list = []
                
                self.logger.info(f"Starting Output Set Creation for {file_path}")
                start = time.time()
                for idx, signal in enumerate(input_signals):
                    quantized_signals = DUT.create_output_signal(signal, real_time)
                    output_signal = quantized_signals.get('quantized_values')
                    output_signal_list.append(output_signal)
                    
                    if idx == 0:
                        if not dictionary_file.exists():
                            match DUT_type:
                                case "nyfr":               
                                    lo_phase_mod_mid = DUT.get_lo_phase_mod_mid()
                                    dictionary = DUT.create_dictionary(lo_phase_mod_mid)
                            np.save(dictionary_file, dictionary)
                            self.logger.info(f"DUT {DUT_type} Dictionary saved to file {dictionary_file}")
                            
                        if not samp_time_file.exists():
                            np.save(samp_time_file, quantized_signals.get('mid_times'))
                            self.logger.info(f"DUT {DUT_type} Sample time array saved to file {samp_time_file}")
                        
                        if not samp_freq_file.exists():
                            np.save(samp_freq_file, quantized_signals.get('sampled_frequency'))
                            self.logger.info(f"DUT {DUT_type} Sample frequency array saved to file {samp_freq_file}")
                            
                        if not wbf_time_file.exists():
                            np.save(wbf_time_file, DUT.get_wbf_time())
                            self.logger.info(f"DUT {DUT_type} Wideband Filter time array saved to file {wbf_time_file}")
                        
                        if not wbf_freq_file.exists():
                            np.save(wbf_freq_file, DUT.get_wbf_freq())
                            self.logger.info(f"DUT {DUT_type} Wideband Filter frequency array saved to file {wbf_freq_file}")
                        
                stop = time.time()
                self.logger.info(f"{len(input_signals)} Signal Output Set Creation Time: {stop - start:.6f} seconds")
                    
                np.save(output_signal_file, np.array(output_signal_list))
                
                self.logger.info(f"Output Set Creation Complete for Input Set {file_path}")
                
        self.logger.info(f"Output Set Creation Complete")

    def create_nyfr_wave_params(self, nyfr):
        self.logger.info(f"Starting NYFR folded wave parameter Creation...")
        
        DUT_type = self.outputset_params.get('DUT_type', None).lower()
        DUT_config_name = self.DUT_config_name
        class_name = nyfr.__class__.__name__.lower()
        config_name = nyfr.get_config_name()
        if class_name != "nyfr":
            self.logger.error(f"{class_name} is not a NYFR object")
            raise ValueError(f"{class_name} is not a NYFR object")
        elif DUT_type != "nyfr":
            self.logger.error(f"Output set parameter {DUT_type} not NYFR")
            raise ValueError(f"Output set parameter {DUT_type} not NYFR")
        elif DUT_config_name != config_name:
            self.logger.error(f"DUT config name {DUT_config_name} does not match NYFR config name {config_name}")
            raise ValueError(f"DUT config name {DUT_config_name} does not match NYFR config name {config_name}")            
        
        LO_params = nyfr.get_lo_params()
        LO_freq = LO_params.get('freq')
        input_dir = self.directories.get('inputs', "Inputs")
        input_signal_wave_params = self.filenames.get('input_wave_params', "wave_params.pkl")
        output_dir = self.directories.get('outputs', "Outputs")
        output_signal_filename = self.filenames.get('output_signal', "signals.npy")
        samp_freq_filename = self.filenames.get('samp_freq', "sampled_freq.npy")
        samp_freq = np.load(output_dir / samp_freq_filename)
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_signal_wave_params):
                stem = file_path.name
                key_part = stem.split(input_signal_wave_params)[0]
                output_signal_file = output_dir / f"{key_part}{output_signal_filename}"
                nyfr_wave_file = output_dir / f"{key_part}{input_signal_wave_params}"
                
                if output_signal_file.exists():
                    with open(file_path, "rb") as f:
                        input_wave_params = pickle.load(f)
                    
                    nyfr_signals = np.load(output_signal_file)
                    nyfr_centered_signals = np.zeros_like(nyfr_signals)
                    nyfr_wave_params = []
                    
                    for idx, nyfr_signal in enumerate(nyfr_signals):
                        nyfr_centered_signal = nyfr_signal - np.mean(nyfr_signal)
                        nyfr_centered_signals[idx] = nyfr_centered_signal
                        nyfr_signal_amp = fftshift(np.abs(fft(nyfr_centered_signal))) / len(samp_freq)
                        nyfr_signal_phase = fftshift(np.angle(fft(nyfr_centered_signal)))
                        input_wave_param = input_wave_params[idx]
                        nyfr_waves = []
                        
                        for input_wave in input_wave_param:
                            nyfr_wave = input_wave
                            input_freq = input_wave.get('freq')
                            folded_freq = np.abs(input_freq - LO_freq * round(input_freq/LO_freq))
                            freq_idx = np.abs(samp_freq - folded_freq).argmin()
                            nyfr_wave['amp'] = 2 * nyfr_signal_amp[freq_idx]
                            nyfr_wave['freq'] = samp_freq[freq_idx]
                            nyfr_wave['phase'] = nyfr_signal_phase[freq_idx]
                            nyfr_waves.append(nyfr_wave)
                        
                        nyfr_wave_params.append(nyfr_waves)
                    
                    np.save(output_signal_file, nyfr_centered_signals)
                    self.logger.info(f"Centered NYFR output file saved to {output_signal_file}")
                    
                    with open(nyfr_wave_file, 'wb') as file:
                        pickle.dump(nyfr_wave_params, file)
                    self.logger.info(f"NYFR folded wave parameter file saved to {nyfr_wave_file}")
                
                else:
                    self.logger.error(f"NYFR output file {output_signal_file} does not exists for input set file {file_path}")
        
        self.logger.info(f"NYFR folded wave parameter creation complete")
        
    
    def create_premultiply_set(self,
                               dictionary_path=None,
                               input_config_name=None,
                               DUT_config_name=None):
        self.logger.info(f"Starting Premultiply Set Creation...")
        
        output_dir = self.directories.get('outputs', "Outputs")
        output_signal_filename = self.filenames.get('output_signal', "time_signals.npy")
        
        if input_config_name is not None:
            self.set_input_config_name(input_config_name)
        if DUT_config_name is not None:
            self.set_DUT_config_name(DUT_config_name)
            
        if dictionary_path is None:
            dictionary_filename = self.filenames.get('dictionary',"dictionary.npy")
            dictionary_path = output_dir / dictionary_filename
        if not dictionary_path.exists():
            self.logger.error("Dictionary File Does Not Exist")
            raise ValueError("Dictionary File Does Not Exist")
        dictionary = np.load(dictionary_path)
            
        premultiply_dir = self.directories.get('premultiply', "Premultiply")
        premultiply_dir.mkdir(parents=True, exist_ok=True)
        premultiply_filename = self.filenames.get('premultiply', "premult_signals.npy")
        
        scale_dict = self.outputset_params.get('scale_dict', 1.0)
        scaled_dictionary = scale_dict * dictionary

        premultiply_signal_list = []
        cp = import_module("cupy")
        Scaled_Dictionary = cp.asarray(scaled_dictionary, dtype=cp.complex64)
        Pinv_Dict = cp.linalg.pinv(Scaled_Dictionary)
        
        for file_path in output_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(output_signal_filename):
                stem = file_path.name
                key_part = stem.split(output_signal_filename)[0]
                premultiply_file = premultiply_dir / f"{key_part}{premultiply_filename}"
                output_signals = np.load(file_path)

                self.logger.info(f"Starting Premultiply Set Creation for {file_path}")
                start = time.time()
                
                for signal in output_signals:
                    Signal = cp.asarray(signal, dtype=cp.complex64)
                    result_gpu = Pinv_Dict @ Signal           # stays on GPU
                    premultiply_signal_list.append(cp.asnumpy(result_gpu))  # move to CPU list
                
                stop = time.time()
                self.logger.info(f"{len(output_signals)} Signal Premultiply Set Creation Time: {stop - start:.6f} seconds")
                
                # Save as NumPy array
                np.save(premultiply_file, np.array(premultiply_signal_list, dtype=np.complex64))
                self.logger.info(f"Premultiply Set Creation Complete for Output Set {file_path}")
                 
        self.logger.info(f"Premultiply Set Creation Complete")

    def create_recovery_set(self,
                            recovery,
                            dictionary_path=None,
                            input_config_name=None,
                            DUT_config_name=None):
        self.logger.info(f"Starting Recovery Set Creation...")
        
        output_dir = self.directories.get('outputs', "Outputs")        
        output_signal_filename = self.filenames.get('output_signal', "signals.npy")
        
        if input_config_name is not None:
            self.set_input_config_name(input_config_name)
        if DUT_config_name is not None:
            self.set_DUT_config_name(DUT_config_name)
            
        if dictionary_path is None:
            output_dirs = self.directories.get('outputs', "Outputs")
            dictionary_filename = self.filenames.get('dictionary',"dictionary.npy")
            dictionary_path = output_dirs / dictionary_filename
        if not dictionary_path.exists():
            self.logger.error("Dictionary File Does Not Exist")
            raise ValueError("Dictionary File Does Not Exist")
        dictionary = np.load(dictionary_path)

        # --- Setup and pre-saves ---
        if recovery is None:
            self.logger.error("Recovery Object not set")
            raise ValueError("Recovery Object Not Set")
        else:
            self.set_recovery_config_name(recovery.get_config_name())
    
        recovery_dirs = self.directories.get('recovery', "Recovery")
        recovery_dirs.mkdir(parents=True, exist_ok=True)
        recovered_signal_filename = self.filenames.get("recovered", "recovered.npy")
        recovery_file = recovery_dirs / f"{key_part}{recovered_signal_filename}"
        
        # --- Config file ---     
        recovery_config_filename = self.filenames.get('recovery_config', "recovery_config.json")
        recovery_config_file = recovery_dirs.parent / recovery_config_filename           
        recovery_params = recovery.get_all_params()
        if not recovery_config_file.exists():
            recovery_config = {
                "config_name": self.config_name,
                "recovery": recovery_params
            }
            save_to_json(recovery_config, recovery_config_file)
            
        recovery_method = recovery_params.get('method')
        
        for file_path in output_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(output_signal_filename):
                output_signals = np.load(file_path)
                # Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(output_signal_filename)[0]
                
                recovered_sig_list = []

                self.logger.info(f"Starting Recovery Set Creation for {file_path}")
                start = time.time() 
                
                for signal in output_signals:
                    recovered_sig_list.append(recovery.recover_signal(signal, dictionary))
                    
                stop = time.time()
                self.logger.info(f"{len(output_signals)} Signal Recovery Set Creation Time: {stop - start:.6f} seconds")
                        
                np.save(recovery_file, np.array(recovered_sig_list))
                self.logger.info(f"Recovery Set Creation Complete for Output Set {file_path} using Recovery Method {recovery_method}")
        
        self.logger.info(f"Recovery Set Creation Complete")
        
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
    
    
    def get_ML_config_name(self):
        return self.ML_config_name

    
    def get_directories(self):
        return self.directories

    
    def get_filenames(self):
        return self.filenames


    def get_log_params(self):
        return self.log_params
    

    def get_directory_params(self):
        return self.directory_params
    
    
    def get_inputset_params(self):
        return self.inputset_params
    
    
    def get_outputset_params(self):
        return self.outputset_params
    
    
    def get_all_params(self):
        dataset_params = {
            "config_name": self.config_name,
            "DUT_config_name": self.DUT_config_name,
            "input_config_name": self.input_config_name,
            "recovery_config_name": self.recovery_config_name,
            "ML_config_name": self.ML_config_name,
            "inputset_params": self.inputset_params,
            "outputset_params": self.outputset_params,
            "directory_params": self.directory_params,
            "log_params": self.log_params,
            "directories": self.directories,
            "filenames": self.filenames
        }
        return dataset_params