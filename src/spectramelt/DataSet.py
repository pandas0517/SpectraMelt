from .utils import (
    load_config_from_json,
    get_logger,
    build_flat_paths,
    flatten_files,
    find_project_root,
    save_to_json,
    fft_encode_signals,
    fft_decode_signals
)
import numpy as np
import pickle
import time
import copy
from importlib import import_module
from scipy.fft import fft, fftshift
import pandas as pd
from pathlib import Path
from .Recovery import VALID_SAVED_FREQ_MODES

class DataSet:
    VALID_DUT_TYPES = {
        "nyfr"
    }
    # Map modes to flattened filename keys
    FREQ_FILE_KEYS = {
        "complex":          "input.freq.signal",
        "mag_ang":          "input.freq.mag_ang_sig",
        "mag_ang_sincos":   "input.freq.mag_ang_sincos_sig",
        "mag":              "input.freq.mag_sig",
        "ang":              "input.freq.ang_sig",
        "real_imag":        "input.freq.real_imag_sig",
        "real":             "input.freq.real_sig",
        "imag":             "input.freq.imag_sig",
    }
    def __init__(self,
                 input_config_name=None,
                 DUT_config_name=None,
                 recovery_config_name=None,
                 ML_config_name=None,
                 dataset_config_name="DataSet_Config_1",
                 seed=None,
                 dataset_params=None,
                 inputset_params=None,
                 outputset_params=None,
                 log_params=None,
                 filenames=None,
                 directory_params=None,
                 dataframe_params=None,
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
            dataset_params['dataframe_params'] = dataframe_params
            dataframe_params['seed'] = seed
            
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
        dataframe_params = dataset_params.get('dataframe_params', None)
        self.input_rng = np.random.default_rng(dataset_params.get('seed', None))
        
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
        self.set_dataframe_params(dataframe_params)

        
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
                "num_recovery_sigs": 100,
                "tones_per_sig": [1],
                "wave_precision": None,
                "saved_freq_modes": []
            }
        saved_freq_modes = inputset_params.get('saved_freq_modes', [])
        if not self.is_valid_saved_freq_mode(saved_freq_modes):
            self.logger.error("Saved Frequency Mode List Contains invalid Mode")
            raise ValueError("Saved Frequency Mode List Contains invalid Mode")
        self.inputset_params = inputset_params
        

    def set_outputset_params(self, outputset_params):
        if outputset_params is None:
            outputset_params = {
                "DUT_type": "NYFR",
                "saved_freq_modes": [],
                "scale_dict": 1.0,
                "decode_to_time": True
            }
        
        DUT_type = outputset_params.get('DUT_type', None)
        if not self.is_valid_dut_type(DUT_type):
            self.logger.error(f"DUT type {DUT_type} not currently valid")
            raise ValueError(f"DUT type {DUT_type} not currently valid")
        
        saved_freq_modes = outputset_params.get('saved_freq_modes', [])
        if not self.is_valid_saved_freq_mode(saved_freq_modes):
            self.logger.error("Saved Frequency Mode List Contains invalid Mode")
            raise ValueError("Saved Frequency Mode List Contains invalid Mode")
                    
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
                "input": {
                    "config": "inputset_config.json",
                    "time_signal": "time_signals.npy",
                    "wave_params": "wave_params.pkl",
                    "freq": {
                        "signal": "freq_signals.npy",
                        "mag_ang_sig": "freq_mag_ang_signals.npy",
                        "mag_ang_sincos_sig": "freq_mag_ang_sincos_signals.npy",
                        "mag_sig": "freq_mag_signals.npy",
                        "ang_sig": "freq_ang_signals.npy",
                        "real_imag_sig": "freq_real_imag_signals.npy",
                        "real_sig": "freq_real_signals.npy",
                        "imag_sig": "freq_imag_signals.npy"
                    }              
                },
                "DUT_config": "DUT_config.json",
                "dictionary": "dictionary.npy",
                "recovery_df": "recovery_df.pkl",
                "recovery_config": "recovery_config.json",
                "ml_model": "ml_model.keras",
                "ml_config": "ml_config.json"
            }
        self.flat_filenames = flatten_files(filenames)
        self.filenames = filenames
        

    def set_dataframe_params(self, dataframe_params=None):
        if dataframe_params is None:
            dataframe_params = {
                "file_path": None,
                "save_as_csv": True,
                "recovery_mag_thresh": 0.5,
                "meta_column_names": {
                    "input_file_name": "str",
                    "recovery_file_name": "str",
                    "input_config_name": "str",
                    "DUT_config_name": "str",
                },
                "signal_column_names": {
                    "num_rec_freq_" : "float64",
                    "num_spur_freq_": "float64",
                    "ave_rec_mag_err_": "float64",
                    "total_input_tones_": "float64",
                    "rec_tone_thresh_": "float64",
                    "ave_rec_mag_": "float64",
                    "max_rec_mag_": "float64",
                    "min_rec_mag_": "float64",
                    "ave_spur_mag_": "float64",
                    "max_spur_mag_": "float64",
                    "min_spur_mag_": "float64"
                }
            }
        self.dataframe_params = dataframe_params

    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def is_valid_saved_freq_mode(self, name) -> bool:
        if isinstance(name, str):
            name = [name]
        return all(n.lower() in VALID_SAVED_FREQ_MODES for n in name)
    

    def is_valid_dut_type(self, dut_type) -> bool:
        """
        Check if a DUT type or list of DUT types is valid.

        Parameters
        ----------
        dut_type : str or list of str
            DUT type(s) to check.

        Returns
        -------
        bool
            True if all DUT types are valid, False otherwise.
        """
        if isinstance(dut_type, str):
            dut_type = [dut_type]
        
        return all(d.lower() in {t.lower() for t in self.VALID_DUT_TYPES} for d in dut_type)
    
        
    def create_input_set(self, input_signal, normalize=False):
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
        inputset_config_filename = self.flat_filenames.get('input.config', "inputset_config.json")
        input_wave_params_filename = self.flat_filenames.get('input.wave_params', "wave_params.pkl")
        input_time_signal_filename = self.flat_filenames.get('input.time_signal', "time_signals.npy")
        saved_freq_modes = self.inputset_params.get('saved_freq_modes', [])

        input_dirs.mkdir(parents=True, exist_ok=True)
        real_time_file = input_dirs / real_time_filename
        real_freq_file = input_dirs / real_freq_filename
        
        real_time = input_signal.get_analog_time()
        real_freq = input_signal.get_analog_frequency()
        if not real_time_file.exists():
            np.save(real_time_file, real_time)
            self.logger.info(f"Real Time Input Signal saved to file {real_time_file}")
        if not real_freq_file.exists():
            np.save(real_freq_file, real_freq)
            self.logger.info(f"Real Frequency Input Signal saved to file {real_freq_file}")
        
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
            self.logger.info(f"Saved Input Set configuration to file {inputset_config_file}")

        # Build discrete frequency grid
        freq_range = input_signal_wave_params.get('freq_range', (100, 1000))
        freq_subset = real_freq[(real_freq >= freq_range[0]) & (real_freq <= freq_range[1])]
        freq_bins = np.copy(freq_subset)

        amp_range = input_signal_wave_params.get('amp_range', (0.1, 1.0))
        phase_range = input_signal_wave_params.get('phase_range', (0, 1))

        num_input_sigs = self.inputset_params.get('num_sigs', 5000)
        num_recovery_sigs = self.inputset_params.get('num_recovery_sigs', 100)
        tones_per_sig = self.inputset_params.get('tones_per_sig', [1])
        wave_precision = self.inputset_params.get('wave_precision', None)
        
        input_signals_time = np.zeros((num_input_sigs, real_time.size))

        # --- Generate all tone sets ---
        for tones in tones_per_sig:
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
                input_signals_time[input_sig] = input_signal_time

            stop = time.time()
            self.logger.info(f"{num_input_sigs} {tones}-Tone Signal Input Set Creation Time: {stop - start:.6f} seconds")

            # --- Save outputs ---
            inputset_wave_params_path = input_dirs / f"{tones}_tone_{input_wave_params_filename}" 
            with open(inputset_wave_params_path, 'wb') as file:
                pickle.dump(wave_param_list, file)
            self.logger.info(f"{tones}-Tone Time Input Set Wave Parameters saved to file {inputset_wave_params_path}")

            inputset_time_path = input_dirs / f"{tones}_tone_{input_time_signal_filename}"
            
            np.save(inputset_time_path, input_signals_time)
            
            self.logger.info(f"{tones}-Tone Time Input Set saved to file {inputset_time_path}")

            # --- After you've built input_signals_freq with FFT results ---
            if saved_freq_modes:
                for mode in saved_freq_modes:

                    if mode not in VALID_SAVED_FREQ_MODES:
                        self.logger.warning(f"Skipping invalid freq mode: {mode}")
                        continue

                    key = self.FREQ_FILE_KEYS[mode]
                    filename = self.flat_filenames.get(key)
                    if not filename:
                        self.logger.error(f"No filename configured for freq mode '{mode}' (key='{key}')")
                        continue
                    
                    arr, scales = fft_encode_signals(input_signals_time, mode,
                                                apply_fftshift=True, normalize=normalize)

                    # --- Save ---                       
                    if normalize:
                        norm_save_path = input_dirs / f"{tones}_tone_norm_{filename}"
                        scale_save_path = input_dirs / f"{tones}_tone_scale_{filename}"
                        np.save(norm_save_path, arr)
                        self.logger.info(f"{tones}-Tone {mode.upper()} normalized frequency set saved to {norm_save_path}")
                        np.save(scale_save_path, scales)
                        self.logger.info(f"{tones}-Tone {mode.upper()} normalized frequency scales set saved to {scale_save_path}")
                    else:
                        save_path = input_dirs / f"{tones}_tone_{filename}"
                        np.save(save_path, arr)
                        self.logger.info(f"{tones}-Tone {mode.upper()} freq set saved to {save_path}")
                    
            input_signals_time[:] = 0

        input_recovery_signals_time = np.zeros((num_recovery_sigs, real_time.size))
        
        if saved_freq_modes:
            input_recovery_signals_freq = np.zeros((num_recovery_sigs, real_time.size), dtype=np.complex128)
       
        # --- Generate all recovery tone sets ---
        for tones in tones_per_sig:
            wave_param_list = [] # reset for this tone set
            start = time.time()
            for input_sig in range(num_recovery_sigs):
                amps = self.input_rng.uniform(amp_range[0], amp_range[1], tones)
                if wave_precision is not None:
                    amps = np.round(amps, wave_precision)

                # Randomly choose unique bins, then scale back to Hz
                freqs = self.input_rng.choice(freq_bins, size=tones, replace=False)

                # Remove chosen bins from freq_bins (in place)
                freq_bins = freq_bins[~np.isin(freq_bins, freqs)]
                if tones > len(freq_bins):
                    self.logger.debug("Ran out of unique frequency bins for input signals. Resetting...")
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
                input_recovery_signals_time[input_sig] = input_signal_time
                
                if saved_freq_modes:
                    input_signal_freq = fft(input_signal_time)
                    input_recovery_signals_freq[input_sig] = input_signal_freq

            stop = time.time()
            self.logger.info(f"{num_recovery_sigs} {tones}-Tone Recovery Signal Input Set Creation Time: {stop - start:.6f} seconds")

            # --- Save outputs ---
            inputset_wave_params_path = input_dirs / f"{tones}_tone_recovery_{input_wave_params_filename}" 
            with open(inputset_wave_params_path, 'wb') as file:
                pickle.dump(wave_param_list, file)
            self.logger.info(f"{tones}-Tone Recovery Time Input Set Wave Parameters saved to file {inputset_wave_params_path}")

            inputset_time_path = input_dirs / f"{tones}_tone_recovery_{input_time_signal_filename}"
            
            np.save(inputset_time_path, input_recovery_signals_time)
            input_recovery_signals_time[:] = 0
            self.logger.info(f"{tones}-Tone Recovery Time Input Set saved to file {inputset_time_path}")

            # --- After you've built input_signals_freq with FFT results ---
            if saved_freq_modes:
                for mode in saved_freq_modes:

                    if mode not in VALID_SAVED_FREQ_MODES:
                        self.logger.warning(f"Skipping invalid freq mode: {mode}")
                        continue

                    key = self.FREQ_FILE_KEYS[mode]
                    filename = self.flat_filenames.get(key)
                    if not filename:
                        self.logger.error(f"No filename configured for freq mode '{mode}' (key='{key}')")
                        continue
                    
                    arr, scales = fft_encode_signals(input_signals_time, mode,
                                                apply_fftshift=True, normalize=normalize)
                    
                    # --- Save ---                       
                    if normalize:
                        norm_save_path = input_dirs / f"{tones}_tone_recovery_norm_{filename}"
                        scale_save_path = input_dirs / f"{tones}_tone_recovery_scale_{filename}"
                        np.save(norm_save_path, arr)
                        self.logger.info(f"{tones}-Tone {mode.upper()} normalized frequency set saved to {norm_save_path}")
                        np.save(scale_save_path, scales)
                        self.logger.info(f"{tones}-Tone {mode.upper()} normalized frequency scales set saved to {scale_save_path}")
                    else:
                        save_path = input_dirs / f"{tones}_tone_recovery_{filename}"
                        np.save(save_path, arr)
                        self.logger.info(f"{tones}-Tone {mode.upper()} freq set saved to {save_path}")
                    
                input_recovery_signals_freq[:] = 0
                
        self.logger.info("All Input Sets Created and Saved\n")


    def create_output_set(self, DUT, input_signal=None, normalize=False):
        self.logger.info(f"Starting Output Set Creation...")
        
        # --- Setup and pre-saves ---
        if input_signal is not None:
            self.set_input_config_name(input_signal.get_config_name())
        
        outputset_params = self.outputset_params
        if DUT is None:
            self.logger.error("DUT Object not set")
            raise ValueError("DUT Object Not Set")
        else:
            outputset_params['DUT_type'] = type(DUT).__name__
            self.set_outputset_params(outputset_params)
            self.set_DUT_config_name(DUT.get_config_name())
        
        input_dir = self.directories.get('inputs', "Inputs")
        input_time_signal_filename = self.flat_filenames.get('input.time_signal', "time_signals.npy")
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
        dictionary_filename = self.filenames.get('dictionary',"dictionary.npy")
        
        samp_time_filename = self.filenames.get('samp_time', "sampled_time.npy")
        samp_freq_filename = self.filenames.get('samp_freq', "sampled_freq.npy")
        
        wbf_time_filename = self.filenames.get('wbf_time', "wbf_time.npy")
        wbf_freq_filename = self.filenames.get('wbf_freq', "wbf_freq.npy")
        
        output_dirs.mkdir(parents=True, exist_ok=True)
        saved_freq_modes = self.outputset_params.get('saved_freq_modes', [])
        dictionary_file = output_dirs / dictionary_filename
        
        samp_time_file = output_dirs / samp_time_filename
        samp_freq_file = output_dirs / samp_freq_filename 
        
        wbf_time_file = output_dirs / wbf_time_filename
        wbf_freq_file = output_dirs / wbf_freq_filename
        
        # --- Config file ---
        DUT_config_file = output_dirs.parent / DUT_config_filename
        DUT_params = DUT.get_all_params()
        DUT_type = outputset_params.get('DUT_type', "nyfr")

        if not DUT_config_file.exists():
            DUT_config = {
                "config_name": self.config_name,
                "output": DUT_params,
                "outputset": outputset_params
            }
            save_to_json(DUT_config, DUT_config_file)
            self.logger.info(f"Saved DUT {DUT_type} configuration to file {DUT_config_file}")
        
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
          
                input_signals = np.load(file_path)
                
                # Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(input_time_signal_filename)[0]
                output_signal_file = output_dirs / f"{key_part}{input_time_signal_filename}"
                wbf_dut_signal_file = output_dirs / f"{key_part}wbf_{input_time_signal_filename}"
                
                dictionary = None
                output_signal_list = []
                wbf_signal_list = []
                self.logger.info(f"Starting Output Set Creation for {file_path}")
                start = time.time()
                for idx, signal in enumerate(input_signals):
                    quantized_signals = DUT.create_output_signal(signal, real_time)
                    wbf_signal = DUT.get_wbf_signal_sub()
                    output_signal = quantized_signals.get('quantized_values')
                    output_signal_list.append(output_signal)
                    wbf_signal_list.append(wbf_signal)

                    if idx == 0:
                        if not dictionary_file.exists():
                            match DUT_type.lower():
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
                self.logger.info(f"Output Set for Input Set {file_path} saved to file {output_signal_file}")

                wbf_signals = np.array(wbf_signal_list)
                np.save(wbf_dut_signal_file, wbf_signals)
                self.logger.info(f"Wideband Filter Set for Input Set {file_path} saved to file {wbf_dut_signal_file}")

                if saved_freq_modes:
                    for mode in saved_freq_modes:

                        if mode not in VALID_SAVED_FREQ_MODES:
                            self.logger.warning(f"Skipping invalid freq mode: {mode}")
                            continue

                        key = self.FREQ_FILE_KEYS[mode]
                        filename = self.flat_filenames.get(key)
                        if not filename:
                            self.logger.error(f"No filename configured for freq mode '{mode}' (key='{key}')")
                            continue

                        arr, scales = fft_encode_signals(wbf_signals, mode, normalize=normalize)

                        # --- Save ---                       
                        if normalize:
                            norm_save_path = output_dirs / f"{key_part}norm_wbf_{filename}"
                            scale_save_path = output_dirs / f"{key_part}scale_wbf_{filename}"
                            np.save(norm_save_path, arr)
                            self.logger.info(f"{key_part}{mode.upper()} wideband filter normalized frequency set saved to {norm_save_path}")
                            np.save(scale_save_path, scales)
                            self.logger.info(f"{key_part}{mode.upper()} wideband filter normalized frequency scales set saved to {scale_save_path}")
                        else:
                            save_path = output_dirs / f"{key_part}wbf_{filename}"
                            np.save(save_path, arr)
                            self.logger.info(f"{key_part}{mode.upper()} wideband filter frequency set saved to {save_path}")
                
        self.logger.info("Output Set Creation Complete\n")


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
        input_wave_params_filename = self.flat_filenames.get('input.wave_params', "wave_params.pkl")
        output_dir = self.directories.get('outputs', "Outputs")
        input_time_signal_filename = self.flat_filenames.get('input.time_signal', "time_signals.npy")
        samp_freq_filename = self.filenames.get('samp_freq', "sampled_freq.npy")
        samp_freq = np.load(output_dir / samp_freq_filename)
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_wave_params_filename):
                stem = file_path.name
                key_part = stem.split(input_wave_params_filename)[0]
                output_signal_file = output_dir / f"{key_part}{input_time_signal_filename}"
                nyfr_wave_file = output_dir / f"{key_part}{input_wave_params_filename}"
                
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
        
        self.logger.info("NYFR folded wave parameter creation complete\n")
        
        
    def create_wbf_wave_params(self):
        self.logger.info(f"Starting Wideband Filter wave parameter Creation...")

        input_dir = self.directories.get('inputs', "Inputs")
        input_time_signal_filename = self.flat_filenames.get('input.time_signal', "time_signals.npy")
        input_wave_params_filename = self.flat_filenames.get('input.wave_params', "wave_params.pkl")
        output_dir = self.directories.get('outputs', "Outputs")
        wbf_freq_filename = self.filenames.get('wbf_freq', "wbf_freq.npy")
        wbf_freq_file = output_dir / wbf_freq_filename
        
        if not wbf_freq_file.exists():
            self.logger.error(f"{wbf_freq_file} does not exist")
            raise ValueError(f"{wbf_freq_file} does not exist")
        wbf_freq = np.load(wbf_freq_file)

        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_wave_params_filename):
                stem = file_path.name
                key_part = stem.split(input_wave_params_filename)[0]
                
                wbf_dut_signal_file = output_dir / f"{key_part}wbf_{input_time_signal_filename}"
                if not wbf_dut_signal_file.exists():
                    self.logger.error(f"{wbf_dut_signal_file} does not exist")
                    raise ValueError(f"{wbf_dut_signal_file} does not exist")
                                    
                wbf_dut_wave_file = output_dir / f"{key_part}{input_wave_params_filename}"

                with open(file_path, "rb") as f:
                    input_wave_params = pickle.load(f)
                
                wbf_time_signals = np.load(wbf_dut_signal_file)
                wbf_freq_signals = fft(wbf_time_signals, axis=1)
                wbf_dut_wave_params = []
                
                for idx, input_wave_param in enumerate(input_wave_params):                   
                    wbf_freq_signal = wbf_freq_signals[idx]
                    wbf_freq_signal_mag = fftshift(np.abs(wbf_freq_signal)) / len(wbf_freq_signal)
                    wbf_freq_signal_phase = fftshift(np.angle(wbf_freq_signal))                    
                    wbf_waves = []
                    
                    for input_wave in input_wave_param:
                        wbf_wave = input_wave
                        input_freq = input_wave.get('freq')
                        freq_idx = np.abs(wbf_freq - input_freq).argmin()
                        wbf_wave['amp'] = 2 * wbf_freq_signal_mag[freq_idx]
                        wbf_wave['freq'] = wbf_freq[freq_idx]
                        wbf_wave['phase'] = wbf_freq_signal_phase[freq_idx]
                        wbf_waves.append(wbf_wave)
                    
                    wbf_dut_wave_params.append(wbf_waves)
                
                with open(wbf_dut_wave_file, 'wb') as file:
                    pickle.dump(wbf_dut_wave_params, file)
                self.logger.info(f"Wideband filtered DUT wave parameter file saved to {wbf_dut_wave_file}")
        
        self.logger.info("Wideband filtered DUT wave parameter creation complete\n")
        
    
    def create_premultiply_set(self,
                               dictionary_path=None,
                               input_config_name=None,
                               DUT_config_name=None,
                               normalize=False):
        self.logger.info(f"Starting Premultiply Set Creation...")
        
        saved_freq_modes = self.outputset_params.get('saved_freq_modes', [])
        output_dir = self.directories.get('outputs', "Outputs")
        input_time_signal_filename = self.flat_filenames.get('input.time_signal', "time_signals.npy")
        
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
        
        scale_dict = self.outputset_params.get('scale_dict', 1.0)
        scaled_dictionary = scale_dict * dictionary

        cp = import_module("cupy")
        Scaled_Dictionary = cp.asarray(scaled_dictionary, dtype=cp.complex64)
        Pinv_Dict = cp.linalg.pinv(Scaled_Dictionary)
        
        for file_path in output_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename) and "wbf" not in file_path.name.lower():
                stem = file_path.name
                key_part = stem.split(input_time_signal_filename)[0]
                
                output_signals = np.load(file_path)

                self.logger.info(f"Starting Premultiply Set Creation for {file_path}")
                start = time.time()
                premultiply_signal_list = []
                
                for signal in output_signals:
                    Signal = cp.asarray(signal, dtype=cp.complex64)
                    result_gpu = Pinv_Dict @ Signal           # stays on GPU
                    premultiply_signal_list.append(cp.asnumpy(result_gpu))  # move to CPU list
                
                stop = time.time()
                self.logger.info(f"{len(output_signals)} Signal Premultiply Set Creation Time: {stop - start:.6f} seconds")
                
                # Save as NumPy array
                premultiply_signals = np.array(premultiply_signal_list, dtype=np.complex64)
                if not saved_freq_modes:
                    premultiply_filename = self.flat_filenames.get('input.freq.signal', "freq_signals.npy")
                    premultiply_file = premultiply_dir / f"{key_part}{premultiply_filename}"
                    np.save(premultiply_file, premultiply_signals)
                    self.logger.info(f"{key_part} premultiply set saved to {premultiply_file}")
                else:
                    for mode in saved_freq_modes:

                        if mode not in VALID_SAVED_FREQ_MODES:
                            self.logger.warning(f"Skipping invalid freq mode: {mode}")
                            continue

                        key = self.FREQ_FILE_KEYS[mode]
                        filename = self.flat_filenames.get(key)
                        if not filename:
                            self.logger.error(f"No filename configured for freq mode '{mode}' (key='{key}')")
                            continue                       

                        arr, scales = fft_encode_signals(premultiply_signals, mode,
                                                         apply_fft=False, normalize=normalize)

                        # --- Save ---
                        if normalize:
                            norm_save_path = premultiply_dir / f"{key_part}norm_{filename}"
                            scale_save_path = premultiply_dir / f"{key_part}scale_{filename}"
                            np.save(norm_save_path, arr)
                            self.logger.info(f"{key_part}{mode.upper()} normalized premultiply set saved to  {norm_save_path}")
                            np.save(scale_save_path, scales)
                            self.logger.info(f"{key_part}{mode.upper()} scales premultiply set saved to {scale_save_path}")
                        else:
                            save_path = premultiply_dir / f"{key_part}{filename}"
                            np.save(save_path, arr)
                            self.logger.info(f"{key_part}{mode.upper()} premultiply set saved to  {save_path}")
                            
                premultiply_signals[:] = 0        
                self.logger.info(f"Premultiply Set Creation Complete for Output Set {file_path}")
                 
        self.logger.info(f"Premultiply Set Creation Complete\n")


    def create_recovery_set(self,
                            recovery,
                            mlp=None,
                            dictionary_path=None,
                            input_config_name=None,
                            DUT_config_name=None):
        self.logger.info(f"Starting Recovery Set Creation...")
        
        output_dir = self.directories.get('outputs', "Outputs")
        premultiply_dir = self.directories.get('premultiply', "Premultiply")       
        input_time_signal_filename = self.flat_filenames.get('input.time_signal', "time_signals.npy")
        
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
        
        # --- Config file ---     
        recovery_config_filename = self.filenames.get('recovery_config', "recovery_config.json")
        recovery_config_file = recovery_dirs.parent / recovery_config_filename         
        all_recovery_params = recovery.get_all_params()
        if not recovery_config_file.exists():
            recovery_config = {
                "config_name": self.config_name,
                "recovery": all_recovery_params
            }
            save_to_json(recovery_config, recovery_config_file)
            self.logger.info(f"Saved recovery configuration file to {recovery_config_file}")
        
        recovery_params = recovery.get_recovery_params()
        recovery_method = recovery_params.get('method').lower()
        saved_freq_modes = self.outputset_params.get('saved_freq_modes', [])
        
        if recovery_method != "mlp":
            for file_path in output_dir.iterdir():
                if (file_path.is_file() and 
                    file_path.name.endswith(input_time_signal_filename) and 
                    "recovery" in file_path.name.lower()):
                    
                    output_signals = np.load(file_path)

                    self.logger.info(f"Starting Recovery Set Creation for {file_path}")
                    start = time.time() 
                    
                    for signal in output_signals:
                            recovered_sig_list.append(recovery.recover_signal(signal, dictionary))
                    stop = time.time()
                    self.logger.info(f"{len(output_signals)} Signal Recovery Set Creation Time: {stop - start:.6f} seconds")
                            
                    np.save(recovery_file, np.array(recovered_sig_list))
                    self.logger.info(f"Recovery Set Creation Complete for Output Set {file_path} using Recovery Method {recovery_method}")
        elif saved_freq_modes:
            ml_models_dir = self.directories.get('ml_models', "ML_Models")
            ml_model_filename = self.flat_filenames.get('ml_model', "ml_model.keras")
            
            for mode in saved_freq_modes:
                key = self.FREQ_FILE_KEYS[mode]
                filename = self.flat_filenames.get(key)
                
                if not filename:
                    self.logger.error(f"No filename configured for freq mode '{mode}' (key='{key}')")
                    continue
                recovery.set_recovery_type(mode)

                if mlp is None:
                    self.logger.error("No MLP object given")
                    raise ValueError("No MLP object given")

                norm_premultiply_h5_file = premultiply_dir / f"{Path(filename).stem}_norm.h5"
                if not norm_premultiply_h5_file.exists():
                    self.logger.error(f"{norm_premultiply_h5_file} file does not exist")
                    raise ValueError(f"{norm_premultiply_h5_file} file does not exist")
                mlp.set_recovery_stats_from_h5(norm_premultiply_h5_file, dataset_name="X")
                    
                norm_output_h5_file = output_dir / f"wbf_{Path(filename).stem}_norm.h5"
                if not norm_output_h5_file.exists():
                    self.logger.error(f"{norm_output_h5_file} file does not exist")
                    raise ValueError(f"{norm_output_h5_file} file does not exist")
                mlp.set_recovery_stats_from_h5(norm_output_h5_file, dataset_name="y")
                
                ml_model_file = ml_models_dir / f"{mode}_{ml_model_filename}"
                if not ml_model_file.exists():
                    self.logger.error(f"{ml_model_file} file does not exist")
                    raise ValueError(f"{ml_model_file} file does not exist")
                mlp.set_model_file_path(ml_model_file)
                mlp_model = mlp.load_model()

                for file_path in premultiply_dir.iterdir():
                    if (file_path.is_file() and 
                        file_path.name.endswith(filename) and 
                        "recovery" in file_path.name.lower()):

                        output_signals = np.load(file_path)
                        # Extract identifying portion (for example, everything up to "signals.npy")
                        stem = file_path.name
                        key_part = stem.split(filename)[0]
                        recovery_file = recovery_dirs / f"{key_part}{filename}"
                        
                        recovered_sig_list = []

                        self.logger.info(f"Starting Recovery Set Creation for {file_path}")
                        start = time.time() 
                        
                        for signal in output_signals:
                                recovered_sig_list.append(recovery.recover_signal(signal, MLP=mlp, mlp_model=mlp_model))

                        stop = time.time()
                        self.logger.info(f"{len(output_signals)} Signal Recovery Set Creation Time: {stop - start:.6f} seconds")
                                
                        np.save(recovery_file, np.array(recovered_sig_list))
                        self.logger.info(f"Recovery Set Creation Complete for Output Set {file_path} using Recovery Method {recovery_method}")
        else:
            self.logger.error(f"Recovery method is {recovery_method} but no save frequency list given")
            raise ValueError(f"Recovery method is {recovery_method} but no save frequency list given")
        
        self.logger.info("Recovery Set Creation Complete\n")


    def decode_complex_sets(self, dir):
        if dir is None:
            self.logger.error("Directory can not be None")
            raise ValueError("Directory can not be None")
        if not dir.is_dir():
            self.logger.error(f"{dir} does not exist")
            raise ValueError(f"{dir} does not exist")

        self.logger.info(f"Decoding complex signals from {dir} into separate magnitude and phase arrays")
        for file in dir.iterdir():
            if ("sincos" in file.name.lower() and
                not file.suffix.lower() == ".h5"):
                complex_npz_filename = file.with_suffix(".npz")
                mag_ang_sincos = np.load(file)
                complex_recovery = fft_decode_signals(mag_ang_sincos)
                complex_mag_recovery = np.abs(complex_recovery)
                complex_phase_recovery = np.angle(complex_recovery)
                np.savez(complex_npz_filename,
                         complex_mag=complex_mag_recovery,
                         complex_phase=complex_phase_recovery,
                         source=str(file),
                         encoding="mag_ang_sincos")
                self.logger.info(f"Converted {file} to {complex_npz_filename}")

        
    def create_recovery_dataframe(self):
        self.logger.info("Creating Dataframe for recovery signals")
        # --- Config file ---
        input_dirs = self.directories.get('inputs', "Inputs")
        inputset_config_filename = self.flat_filenames.get('input.config', "inputset_config.json")
        inputset_config_file = input_dirs.parent / inputset_config_filename
        
        recovery_dirs = self.directories.get('recovery', "Recovery")
        recovery_df_filename = self.dataframe_params.get('file_path', "recovery_df.pkl")
        recovery_df_file_path = recovery_dirs / recovery_df_filename

        if recovery_df_file_path.exists():
            self.logger.warning(f"{recovery_df_file_path} exists.  Will be overwritten")
            
        if not inputset_config_file.exists():
            self.logger.error(f"{inputset_config_file} does not exist")
            raise ValueError(f"{inputset_config_file} does not exist")
        else:
            inputset_config = load_config_from_json(inputset_config_file)
                        
        meta_column_names = self.dataframe_params.get('meta_column_names')
        signal_column_names = self.dataframe_params.get('signal_column_names')
        input_config = inputset_config.get('inputset')
        num_recovery_sigs = input_config.get('num_recovery_sigs')
        
        # Build the master column dictionary
        full_column_dict = dict(meta_column_names)   # start with static columns

        for sig in range(num_recovery_sigs):
            for prefix, dtype in signal_column_names.items():
                full_column_dict[f"{prefix}{sig}"] = dtype

        # Create empty DataFrame
        recovery_df = pd.DataFrame({
            col: pd.Series(dtype=dtype) for col, dtype in full_column_dict.items()
        })

        recovery_df.to_pickle(recovery_df_file_path)
        self.logger.info(f"Saved Dataframe to {recovery_df}")
        
        save_as_csv = self.dataframe_params.get('save_as_csv', True)
        if save_as_csv:
            recovery_df_file_path_csv = recovery_df_file_path.with_suffix(".csv")
            recovery_df.to_csv(recovery_df_file_path_csv, index=False)
            self.logger.info(f"Saved CSV Dataframe to {recovery_df_file_path_csv}")
            
            
    def set_recovery_dataframe(self):
        input_dir = self.directories.get('inputs', "Inputs")
        inputset_config_filename = self.flat_filenames.get('input.config', "inputset_config.json")
        input_wave_params_filename = self.flat_filenames.get('input.wave_params', "wave_params.pkl")
        inputset_config_file = input_dir.parent / inputset_config_filename
        
        output_dir = self.directories.get('outputs', "Outputs")
        wbf_freq_filename = self.filenames.get('wbf_freq', "wbf_freq.npy")
        wbf_freq_file = output_dir / wbf_freq_filename

        DUT_config_filename = self.filenames.get('DUT_config', "DUT_config.json")
        DUT_config_file = output_dir.parent / DUT_config_filename
        
        recovery_dir = self.directories.get('recovery', "Recovery")
        recovery_df_filename = self.dataframe_params.get('file_path', "recovery_df.pkl")
        recovery_df_file_path = recovery_dir / recovery_df_filename

        if not recovery_df_file_path.exists():
            self.logger.error(f"{recovery_df_file_path} does not exist.")
            raise ValueError(f"{recovery_df_file_path} does not exist.")
        else:
            recovery_df = pd.read_pickle(recovery_df_file_path)

        if not inputset_config_file.exists():
            self.logger.error(f"{inputset_config_file} does not exist")
            raise ValueError(f"{inputset_config_file} does not exist")
        else:
            inputset_config = load_config_from_json(inputset_config_file)
        
        inputset_config_name = inputset_config.get('config_name')
        num_recovery_sigs = inputset_config.get('num_recovery_sigs')

        if not DUT_config_file.exists():
            self.logger.error(f"{DUT_config_file} does not exist")
            raise ValueError(f"{DUT_config_file} does not exist")
        else:
            DUT_config = load_config_from_json(DUT_config_file)
        
        DUT_config_name = DUT_config.get('config_name')
        
        recovery_mag_threshold = self.dataframe_params.get('recovery_mag_thresh', 0.5)
        saved_freq_modes = self.inputset_params.get('saved_freq_modes', [])

        #Need to add support for real_imag, real, and imag modes in the future
        unsupported_freq_modes = {"real_imag", "real", "imag"}
        if any(t in saved_freq_modes for t in unsupported_freq_modes):
            self.logger.warning("Unsupported frequency modes found. Removing")
            saved_freq_modes = [x for x in saved_freq_modes if x not in unsupported_freq_modes]
            
        for mode in saved_freq_modes:
            key = self.FREQ_FILE_KEYS[mode]
            filename = self.flat_filenames.get(key)
            if not filename:
                self.logger.error(f"No filename configured for freq mode '{mode}' (key='{key}')")
                continue
            
            recovery_dict = {p.name: p for p in recovery_dir.iterdir()
                            if p.is_file() and p.name.endswith(filename)
                            and "recovery" in p.name.lower()}

            input_dict = {p.name: p for p in input_dir.iterdir()
                        if p.is_file() and p.name.endswith(filename)
                        and "recovery" in p.name.lower()}
            
            wbf_wave_dict = {p.name: p for p in output_dir.iterdir()
                        if p.is_file() and p.name.endswith(input_wave_params_filename)
                        and "recovery" in p.name.lower()}
            
            wbf_freq_sig_dict = {p.name: p for p in output_dir.iterdir()
                        if p.is_file() and p.name.endswith(filename)
                        and "recovery" in p.name.lower()}
            
            missing_in_input = recovery_dict.keys() - input_dict.keys()
            missing_in_recovery = input_dict.keys() - recovery_dict.keys()

            if missing_in_input:
                self.logger.error("Files missing in input:", missing_in_input)
                raise ValueError("Files missing in input:", missing_in_input)

            if missing_in_recovery:
                self.logger.error("Files missing in recovery:", missing_in_recovery)
                raise ValueError("Files missing in recovery:", missing_in_recovery)           
            
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
                                    
                for idx,rec_sig in enumerate(recovered_signals):
                    meta_data = self.__create_meta_data_dictionary(idx)
                        
                    input_sig_xf = fft(input_sig)
                    input_sig_tones = np.where(abs(input_sig_xf) > input_tone_thresh)[0]
                    input_tone_mag = np.abs(input_sig_xf)
                    rec_sig_tones = np.where(abs(rec_sig) > recovery_mag_threshold)[0]
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
                    meta_data['rec_tone_thresh']['value'] = recovery_mag_threshold
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


    def get_flat_filenames(self):
        return self.flat_filenames
    
    
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
    
    
    def get_valid_saved_freq_modes(cls):
        return VALID_SAVED_FREQ_MODES
    
    
    def get_valid_dut_types(cls):
        return cls.VALID_DUT_TYPES
    
    
    def get_freq_file_keys(cls):
        return cls.FREQ_FILE_KEYS
    
    
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
            "filenames": self.filenames,
            "filenames_flat": self.filenames_flat
        }
        return dataset_params