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
from scipy.fft import fft, fftshift, ifftshift
import pandas as pd
from pathlib import Path
import platform
import tempfile
import os
from zipfile import ZipFile, ZIP_DEFLATED
from .Recovery import VALID_SAVED_FREQ_MODES

class DataSet:
    VALID_DUT_TYPES = {
        "nyfr"
    }
    def __init__(self,
                 input_config_name=None,
                 DUT_config_name=None,
                 recovery_config_name=None,
                 ML_config_name=None,
                 dataset_config_name="DataSet_Config_1",
                 seed=None,
                 freq_modes=None,
                 dataset_params=None,
                 inputset_params=None,
                 outputset_params=None,
                 premultiply_params=None,
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
            dataframe_params['freq_modes'] = freq_modes
            dataframe_params['premultiply_params'] = premultiply_params
            
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
        freq_modes = dataset_params.get('freq_modes', None)
        premultiply_params = dataset_params.get('premultiply_params', None)
        
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
        self.set_freq_modes(freq_modes)
        self.set_premultiply_params(premultiply_params)

        
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
                "num_sigs": 1000,
                "num_recovery_sigs": 100,
                "tones_per_sig": [1],
                "wave_precision": None,
                "normalize": True,
                "fft_shift": True,
                "overwrite": True
            }

        self.inputset_params = inputset_params
        

    def set_outputset_params(self, outputset_params):
        if outputset_params is None:
            outputset_params = {
                "DUT_type": "NYFR",
                "scale_dict": 1.0,
                "decode_to_time": True,
                "normalize": True,
                "fft_shift": True,
                "normalize_wbf": True,
                "fft_shift_wbf": False,
                "overwrite": True
            }
        
        DUT_type = outputset_params.get('DUT_type', None)
        if not self.is_valid_dut_type(DUT_type):
            self.logger.error(f"DUT type {DUT_type} not currently valid")
            raise ValueError(f"DUT type {DUT_type} not currently valid")
                    
        self.outputset_params = outputset_params


    def set_premultiply_params(self, premultiply_params):
        if premultiply_params is None:
            premultiply_params = {
                "normalize": True,
                "apply_fft": False,
                "fft_shift": True,
                "overwrite": True
            }
                    
        self.premultiply_params = premultiply_params


    def set_freq_modes(self, freq_modes):
        if freq_modes is None:
            freq_modes = {
                "input": ["mag",
                        "ang",
                        "real",
                        "imag"],
                "output": ["mag",
                        "ang",
                        "real",
                        "imag"],
                "wideband": ["mag",
                            "ang",
                            "real",
                            "imag"],
                "mlp": ["mag"],
                "recovery": ["mag"] 
            }
        
        for freq_mode, freq_mode_list in freq_modes.items():
            valid_modes, removed_modes = self.filter_valid_names(freq_mode_list)
            freq_modes[freq_mode] = valid_modes
            if removed_modes:
                self.logger.warning(f"Invalid modes removed from {freq_mode} frequency mode list: {removed_modes}")

        self.freq_modes = freq_modes
        

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
        self.internal_directory_params['tail']['wideband'] = [self.input_config_name,
                                                              DUT_config_name,
                                                              self.directory_params['tail']['outputs'],
                                                              self.directory_params['tail']['wideband']]
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
        """
        Set directory parameters, automatically selecting linux/windows base if available.
        """
        # Default structure
        if directory_params is None:
            directory_params = {}
        directory_params.setdefault('dataset_dir', "Data_Set")
        directory_params.setdefault('paths', [
            "inputs",
            "outputs",
            "premultiply",
            "wideband",
            "recovery",
            "ml_models"
        ])
        directory_params.setdefault('base', {key: None for key in directory_params['paths']})
        directory_params.setdefault('tail', {
            "inputs": "Inputs",
            "outputs": "Outputs",
            "premultiply": "Premultiply",
            "wideband": "Wideband",
            "recovery": "Recovery",
            "ml_models": "ML_Models"
        })

        # OS base paths if provided
        linux_base = directory_params.get("linux_base")
        windows_base = directory_params.get("windows_base")

        if linux_base or windows_base:
            # Detect OS
            system = platform.system().lower()
            os_base = linux_base if "linux" in system else windows_base

            # Assign only unset paths
            for key in directory_params['paths']:
                if directory_params['base'].get(key) is None:
                    directory_params['base'][key] = os_base.get(key, find_project_root())
        else:
            # No OS-specific base provided → assign unset paths to project root
            for key in directory_params['paths']:
                if directory_params['base'].get(key) is None:
                    directory_params['base'][key] = find_project_root()

        self.internal_directory_params = copy.deepcopy(directory_params)
        self.directory_params = directory_params           
        

    def set_filenames(self, filenames=None):
        if filenames is None:
            filenames = {
                "real_time_freq": "real_time_freq.npz",
                "wbf_time_freq": "wbf_time_freq.npz",
                "samp_time_freq": "sampled_time_freq.npz",
                "time_signals": "time_signals.npy",
                "wave_params": "wave_params.pkl",
                "all_freq_signals": "freq_signals.npz",
                "freq_signals": "freq_signals.npy",
                "input_config": "inputset_config.json",
                "recovery_config": "recovery_config.json",
                "ml_config": "ml_config.json",
                "ml_model": "ml_model.keras",
                "DUT_config": "DUT_config.json",
                "dictionary": "dictionary.npy",
                "recovery_df": "recovery_df.pkl"
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
    
    def filter_valid_names(self, names, valid_set=None):
        if valid_set is None:
            valid_set = VALID_SAVED_FREQ_MODES

        # Allow a single string or a list
        if isinstance(names, str):
            names = [names]

        valid = []
        removed = []

        for n in names:
            if n.lower() in valid_set:
                valid.append(n)
            else:
                removed.append(n)

        return valid, removed


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
    
        
    def create_input_set(self,
                         input_signal,
                         normalize=None,
                         fft_shift=None,
                         overwrite=None):
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

        if normalize is None:
            normalize = self.inputset_params.get('normalize', False)

        if overwrite is None:
            overwrite = self.inputset_params.get('overwrite', False)

        if fft_shift is None:
            fft_shift = self.inputset_params.get('fft_shift', False)   
            
        input_dirs = self.directories.get('inputs', "Inputs")
        real_time_freq_filename = self.filenames.get("real_time_freq", "real_time_freq.npz")
        inputset_config_filename = self.filenames.get('input_config', "inputset_config.json")
        input_wave_params_filename = self.filenames.get('wave_params', "wave_params.pkl")
        input_time_signals_filename = self.filenames.get('time_signals', "time_signals.npy")
        input_freq_signals_filename = self.filenames.get('freq_signals', "freq_signals.npz")
        input_freq_modes = self.freq_modes.get('input', [])

        input_dirs.mkdir(parents=True, exist_ok=True)
        real_time_freq_file = input_dirs / real_time_freq_filename
        
        real_time = input_signal.get_analog_time()
        real_freq = input_signal.get_analog_frequency()
        if not real_time_freq_file.exists() or overwrite:
            np.savez(real_time_freq_file,
                     time=real_time,
                     freq=real_freq)
            self.logger.info(f"Real Time and Frequency Signal saved to file {real_time_freq_file}")
        
        # --- Config file ---
        inputset_config_file = input_dirs.parent / inputset_config_filename
        input_signal_params = input_signal.get_all_params()
        input_signal_wave_params = input_signal_params.get('wave_params', None)
        if input_signal_wave_params is None:
            self.logger.error("Input Signal Wave Parameters not set")
            raise ValueError("Input Signal Wave Parameters Not Set")
        
        if not inputset_config_file.exists() or overwrite:
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

        num_input_sigs = self.inputset_params.get('num_sigs', 1000)
        num_recovery_sigs = self.inputset_params.get('num_recovery_sigs', 100)
        tones_per_sig = self.inputset_params.get('tones_per_sig', [1])
        wave_precision = self.inputset_params.get('wave_precision', None)

        # --- Generate all tone sets ---
        for tones in tones_per_sig:
            all_inputset_signals = {
                "dataset": {
                    "time_path": input_dirs / f"{tones}_tone_{input_time_signals_filename}",
                    "freq_path": input_dirs / f"{tones}_tone_{input_freq_signals_filename}",
                    "time_set": [],
                    "wave_path": input_dirs / f"{tones}_tone_{input_wave_params_filename}",
                    "wave_set": []
                },
                "recovery": {
                    "time_path": input_dirs / f"{tones}_tone_recovery_{input_time_signals_filename}",
                    "freq_path": input_dirs / f"{tones}_tone_recovery_{input_freq_signals_filename}",
                    "time_set": [],
                    "wave_path": input_dirs / f"{tones}_tone_recovery_{input_wave_params_filename}",
                    "wave_set": []
                }
            }
            inputset_type = "dataset"
            start = time.time()
            for input_sig in range(num_input_sigs + num_recovery_sigs):
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
                    {
                        "amp": float(amps[i]),
                        "freq": float(freqs[i]),
                        "phase": float(phases[i]),
                        "real": None,
                        "imag": None
                    }
                    for i in range(tones)
                ]

                input_signal_wave_params["waves"] = wave
                input_signal.set_wave_params(input_signal_wave_params)
                input_signal.create_input_signal()

                
                if input_sig == num_input_sigs:
                    inputset_type = "recovery"
                
                all_inputset_signals[inputset_type]["wave_set"].append(wave)
                all_inputset_signals[inputset_type]["time_set"].append(input_signal.get_input_signal())

            stop = time.time()
            self.logger.info(f"{num_input_sigs} {tones}-Tone Signal Input Set Creation Time: {stop - start:.6f} seconds")

            # --- Save outputs ---
            for set_info in all_inputset_signals.values():
                if not set_info.get('wave_path').exists() or overwrite:
                    with open(set_info.get('wave_path'), 'wb') as file:
                        pickle.dump(set_info.get('wave_set'), file)
                    self.logger.info(f"{tones}-Tone Time Input Set Wave Parameters saved to file {set_info.get('wave_path')}")

                time_set = np.array(set_info.get('time_set'))
                if not set_info.get('time_path').exists() or overwrite:
                    np.save(set_info.get('time_path'), time_set)
                    self.logger.info(f"{tones}-Tone Time Input Set saved to file {set_info.get('time_path')}")

                temp_arr = {}
                if input_freq_modes:
                    for mode in input_freq_modes:
                        arr = fft_encode_signals(time_set, mode,
                                                 apply_fftshift=fft_shift,
                                                 normalize=normalize)
                        fd, path_arr = tempfile.mkstemp(suffix=".npy")
                        os.close(fd)
                        np.save(path_arr, arr)
                        temp_arr[mode] = path_arr                    

                    with ZipFile(set_info.get('freq_path'), 'w', ZIP_DEFLATED) as zf:
                        for name, path in temp_arr.items():
                            zf.write(path, arcname=f"{name}.npy")
                    self.logger.info(f"{tones}-Tone frequency set saved to {set_info.get('freq_path')}")

                    for path in temp_arr.values():
                        os.remove(path)                   
                              
        self.logger.info("All Input Sets Created and Saved\n")


    def create_output_set(self, DUT,
                          input_signal=None,
                          normalize=None,
                          fft_shift=None,
                          normalize_wbf=None,
                          fft_shift_wbf=None,
                          overwrite=None):
        self.logger.info(f"Starting Output Set Creation...")
        
        # --- Setup and pre-saves ---
        if input_signal is not None:
            self.set_input_config_name(input_signal.get_config_name())
            
        if normalize is None:
            normalize = self.outputset_params.get('normalize', False)

        if fft_shift is None:
            fft_shift = self.outputset_params.get('fft_shift', False)
            
        if normalize_wbf is None:
            normalize_wbf = self.outputset_params.get('normalize_wbf', False)

        if fft_shift_wbf is None:
            fft_shift_wbf = self.outputset_params.get('fft_shift_wbf', False)
            
        if overwrite is None:
            overwrite = self.outputset_params.get('overwrite', False)
        
        outputset_params = self.outputset_params
        if DUT is None:
            self.logger.error("DUT Object not set")
            raise ValueError("DUT Object Not Set")
        else:
            outputset_params['DUT_type'] = type(DUT).__name__
            self.set_outputset_params(outputset_params)
            self.set_DUT_config_name(DUT.get_config_name())
        
        input_dir = self.directories.get('inputs', "Inputs")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        freq_signals_filename = self.filenames.get('freq_signals', "freq_signals.npz")
        
        real_time_freq_filename = self.filenames.get('real_time_freq', "real_time_freq.npz")
        real_time_freq_file = input_dir / real_time_freq_filename
        if real_time_freq_file.exists():
            real_time_freq = np.load(real_time_freq_file)
            real_time = real_time_freq["time"]
        elif input_signal is not None:
            real_time = input_signal.get_analog_time()
        else:
            self.logger.error("No time file found and Input Signal Object not set")
            raise ValueError("No time file found and Input Signal Object Not Set")
        
        output_dirs = self.directories.get('outputs', "Outputs")
        output_dirs.mkdir(parents=True, exist_ok=True)
        wideband_dir = self.directories.get('wideband', "Wideband")
        wideband_dir.mkdir(parents=True, exist_ok=True)
        
        dictionary_filename = self.filenames.get('dictionary',"dictionary.npy")
        dictionary_file = output_dirs / dictionary_filename
        
        samp_time_freq_filename = self.filenames.get('samp_time_freq', "sampled_time_freq.npz")
        wbf_time_freq_filename = self.filenames.get('wbf_time_freq', "wbf_time_freq.npz")      
        samp_time_freq_file = output_dirs / samp_time_freq_filename
        wbf_time_freq_file = wideband_dir / wbf_time_freq_filename
        
        # --- Config file ---
        DUT_config_filename = self.filenames.get('DUT_config', "DUT_config.json")
        DUT_config_file = output_dirs.parent / DUT_config_filename
        DUT_params = DUT.get_all_params()
        DUT_type = outputset_params.get('DUT_type', "nyfr")

        if not DUT_config_file.exists() or overwrite:
            DUT_config = {
                "config_name": self.config_name,
                "output": DUT_params,
                "outputset": outputset_params
            }
            save_to_json(DUT_config, DUT_config_file)
            self.logger.info(f"Saved DUT {DUT_type} configuration to file {DUT_config_file}")

        output_freq_modes = self.freq_modes.get('output', [])
        wideband_freq_modes = self.freq_modes.get('wideband', [])
        
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                input_signals = np.load(file_path)
                
                # Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(input_time_signal_filename)[0]
                output_signal_file = output_dirs / f"{key_part}{input_time_signal_filename}"
                output_freq_signals_file = output_dirs / f"{key_part}{freq_signals_filename}"
                wbf_dut_signal_file = wideband_dir / f"{key_part}{input_time_signal_filename}"
                wbf_freq_signals_file = wideband_dir / f"{key_part}{freq_signals_filename}"
                
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
                        if not dictionary_file.exists() or overwrite:
                            match DUT_type.lower():
                                case "nyfr":               
                                    lo_phase_mod_mid = DUT.get_lo_phase_mod_mid()
                                    dictionary = DUT.create_dictionary(lo_phase_mod_mid)
                            np.save(dictionary_file, dictionary)
                            self.logger.info(f"DUT {DUT_type} Dictionary saved to file {dictionary_file}")
                            
                        if not samp_time_freq_file.exists() or overwrite:
                            np.savez(samp_time_freq_file,
                                     time=quantized_signals.get('mid_times'),
                                     freq=quantized_signals.get('sampled_frequency'))
                            self.logger.info(f"DUT {DUT_type} sample time and frequency array saved to file {samp_time_freq_file}")
                            
                        if not wbf_time_freq_file.exists() or overwrite:
                            np.savez(wbf_time_freq_file,
                                     time=DUT.get_wbf_time(),
                                     freq=DUT.get_wbf_freq())
                            self.logger.info(f"DUT {DUT_type} Wideband Filter time and frequency array saved to file {wbf_time_freq_file}")
                        
                stop = time.time()
                self.logger.info(f"{len(input_signals)} Signal Output Set Creation Time: {stop - start:.6f} seconds")
                
                output_signals = np.array(output_signal_list)

                np.save(output_signal_file, output_signals)
                self.logger.info(f"Output Set for Input Set {file_path} saved to file {output_signal_file}")
                
                temp_arr = {}
                if output_freq_modes:
                    for mode in output_freq_modes:
                        arr = fft_encode_signals(output_signals, mode,
                                                 apply_fftshift=fft_shift,
                                                 normalize=normalize)
                        fd, path_arr = tempfile.mkstemp(suffix=".npy")
                        os.close(fd)
                        np.save(path_arr, arr)
                        temp_arr[mode] = path_arr                    

                    with ZipFile(output_freq_signals_file, 'w', ZIP_DEFLATED) as zf:
                        for name, path in temp_arr.items():
                            zf.write(path, arcname=f"{name}.npy")
                    self.logger.info(f"{key_part[:-1]} output frequency set saved to {output_freq_signals_file}")

                    for path in temp_arr.values():
                        os.remove(path) 

                wbf_signals = np.array(wbf_signal_list)

                np.save(wbf_dut_signal_file, wbf_signals)
                self.logger.info(f"Wideband Filter Set for Input Set {file_path} saved to file {wbf_dut_signal_file}")
                
                temp_arr = {}
                if wideband_freq_modes:
                    for mode in wideband_freq_modes:
                        arr = fft_encode_signals(wbf_signals, mode,
                                                 apply_fftshift=fft_shift_wbf,
                                                 normalize=normalize_wbf)
                        fd, path_arr = tempfile.mkstemp(suffix=".npy")
                        os.close(fd)
                        np.save(path_arr, arr)
                        temp_arr[mode] = path_arr                    

                    with ZipFile(wbf_freq_signals_file, 'w', ZIP_DEFLATED) as zf:
                        for name, path in temp_arr.items():
                            zf.write(path, arcname=f"{name}.npy")
                    self.logger.info(f"{key_part[:-1]} wideband filter frequency set saved to {wbf_freq_signals_file}")

                    for path in temp_arr.values():
                        os.remove(path) 
                
        self.logger.info("Output Set Creation Complete\n")


    def update_input_wave_params(self):
        self.logger.info(f"Updating input wave parameters...")
        input_dir = self.directories.get('inputs', "Inputs")
        input_wave_params_filename = self.filenames.get('wave_params', "wave_params.pkl")
        input_freq_signals_filename = self.filenames.get('freq_signals', "freq_signals.npz")
        
        real_time_freq_filename = self.filenames.get('real_time_freq', "real_time_freq.npz")
        real_time_freq_file = input_dir / real_time_freq_filename
        
        if real_time_freq_file.exists():
            real_time_freq = np.load(real_time_freq_file)
            real_freq = real_time_freq["freq"]
        else:
            self.logger.error("No time_frequency file found")
            raise ValueError("No time_frequency file found")
        
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_freq_signals_filename):
                stem = str(file_path)
                key_part = stem.split(input_freq_signals_filename)[0]
                wave_params_filename = Path(f"{key_part}{input_wave_params_filename}")

                if wave_params_filename.exists():
                    with open(wave_params_filename, "rb") as f:
                        input_wave_params = pickle.load(f)
                else:
                    self.logger.error("No matching wave parameter file found")
                    raise ValueError("No matching wave parameter file found")                    

                input_signals = np.load(file_path)
                modes = ["real", "imag"]
                update_mag_ang = True
                # Precompute constants once
                PHASE_SHIFT = -1.5*np.pi + np.pi
                # Build a dict for O(1) lookup instead of np.where every time
                freq_to_index = {f: i for i, f in enumerate(real_freq)}

                for mode in modes:
                    for idx, signal in enumerate(input_signals[mode]):
                        wave_param = input_wave_params[idx]

                        for wave in wave_param:
                            if update_mag_ang:
                                wave["amp"] *= 0.5  # amp = amp / 2

                                # Vectorized phase adjustment (but applied per item)
                                wave["phase"] = ((wave["phase"] + PHASE_SHIFT) % (2*np.pi)) - np.pi

                            index = freq_to_index[wave["freq"]]  # O(1) lookup
                            wave[mode] = signal[index]

                        input_wave_params[idx] = wave_param
                        
                    update_mag_ang = False
                
                with open(wave_params_filename, 'wb') as file:
                    pickle.dump(input_wave_params, file)
                self.logger.info(f"Updated input wave parameter file saved to {wave_params_filename}")

        self.logger.info(f"Completed updating input wave parameters...")
            

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
        input_wave_params_filename = self.filenames.get('wave_params', "wave_params.pkl")
        output_dir = self.directories.get('outputs', "Outputs")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        samp_time_freq_filename = self.filenames.get('samp_time_freq', "sampled_time_freq.npz")
        samp_time_freq = np.load(output_dir / samp_time_freq_filename)
        samp_freq = samp_time_freq["freq"]
        N = len(samp_freq)
        
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_wave_params_filename):
                stem = file_path.name
                key_part = stem.split(input_wave_params_filename)[0]
                output_signal_file = output_dir / f"{key_part}{input_time_signal_filename}"
                centered_output_signal_file = output_dir / f"{key_part}centered_{input_time_signal_filename}"
                nyfr_wave_file = output_dir / f"{key_part}{input_wave_params_filename}"
                
                if output_signal_file.exists():
                    with open(file_path, "rb") as f:
                        input_wave_params = pickle.load(f)
                    
                    nyfr_signals = np.load(output_signal_file)
                    nyfr_centered_signals = nyfr_signals - np.mean(nyfr_signals, axis=1, keepdims=True)
                    nyfr_freq_signals = fft(nyfr_centered_signals, axis=1)
                    nyfr_signals_mag = fftshift(np.abs(nyfr_freq_signals), axes=1) / N
                    nyfr_signals_phase = fftshift(np.angle(nyfr_freq_signals), axes=1)
                    nyfr_signals_real = fftshift(np.real(nyfr_freq_signals), axes=1) / N
                    nyfr_signals_imag = fftshift(np.imag(nyfr_freq_signals), axes=1) / N

                    nyfr_wave_params = []
                    
                    for idx, input_wave_param in enumerate(input_wave_params):
                        nyfr_signal_mag = nyfr_signals_mag[idx]
                        nyfr_signal_phase = nyfr_signals_phase[idx]
                        nyfr_signal_real = nyfr_signals_real[idx]
                        nyfr_signal_imag = nyfr_signals_imag[idx]                    
                        nyfr_waves = []
                        
                        for input_wave in input_wave_param:
                            nyfr_wave = input_wave
                            input_freq = input_wave.get('freq')
                            folded_freq = np.abs(input_freq - LO_freq * round(input_freq/LO_freq))
                            freq_idx = np.abs(samp_freq - folded_freq).argmin()
                            nyfr_wave['amp'] = nyfr_signal_mag[freq_idx]
                            nyfr_wave['freq'] = samp_freq[freq_idx]
                            nyfr_wave['phase'] = nyfr_signal_phase[freq_idx]
                            nyfr_wave['real'] = nyfr_signal_real[freq_idx]
                            nyfr_wave['imag'] = nyfr_signal_imag[freq_idx]
                            nyfr_waves.append(nyfr_wave)
                        
                        nyfr_wave_params.append(nyfr_waves)
                    
                    np.save(centered_output_signal_file, nyfr_centered_signals)
                    self.logger.info(f"Centered NYFR output file saved to {centered_output_signal_file}")
                    
                    with open(nyfr_wave_file, 'wb') as file:
                        pickle.dump(nyfr_wave_params, file)
                    self.logger.info(f"NYFR folded wave parameter file saved to {nyfr_wave_file}")
                
                else:
                    self.logger.error(f"NYFR output file {output_signal_file} does not exists for input set file {file_path}")
        
        self.logger.info("NYFR folded wave parameter creation complete\n")
        
        
    def create_wbf_wave_params(self):
        self.logger.info(f"Starting Wideband Filter wave parameter Creation...")

        input_dir = self.directories.get('inputs', "Inputs")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        input_wave_params_filename = self.filenames.get('wave_params', "wave_params.pkl")
        wideband_dir = self.directories.get('wideband', "Wideband")

        wbf_time_freq_filename = self.filenames.get('wbf_time_freq', "wbf_time_freq.npz")      
        wbf_time_freq_file = wideband_dir / wbf_time_freq_filename
        
        if not wbf_time_freq_file.exists():
            self.logger.error(f"{wbf_time_freq_file} does not exist")
            raise ValueError(f"{wbf_time_freq_file} does not exist")
        wbf_time_freq = np.load(wbf_time_freq_file)
        wbf_freq = wbf_time_freq["freq"]
        N = len(wbf_freq)

        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_wave_params_filename):
                stem = file_path.name
                key_part = stem.split(input_wave_params_filename)[0]
                
                wbf_dut_signal_file = wideband_dir / f"{key_part}{input_time_signal_filename}"
                if wbf_dut_signal_file.exists():
                    wbf_dut_wave_file = wideband_dir / f"{key_part}{input_wave_params_filename}"

                    with open(file_path, "rb") as f:
                        input_wave_params = pickle.load(f)
                    
                    wbf_time_signals = np.load(wbf_dut_signal_file)
                    wbf_freq_signals = fft(wbf_time_signals, axis=1)
                    wbf_freq_signals_mag = fftshift(np.abs(wbf_freq_signals), axes=1) / N
                    wbf_freq_signals_phase = fftshift(np.angle(wbf_freq_signals), axes=1)
                    wbf_freq_signals_real = fftshift(np.real(wbf_freq_signals), axes=1) / N
                    wbf_freq_signals_imag = fftshift(np.imag(wbf_freq_signals), axes=1) / N
                    wbf_dut_wave_params = []
                    
                    for idx, input_wave_param in enumerate(input_wave_params):                   
                        wbf_freq_signal_mag = wbf_freq_signals_mag[idx]
                        wbf_freq_signal_phase = wbf_freq_signals_phase[idx]
                        wbf_freq_signal_real = wbf_freq_signals_real[idx]
                        wbf_freq_signal_imag = wbf_freq_signals_imag[idx]

                        wbf_waves = []
                        
                        for input_wave in input_wave_param:
                            wbf_wave = input_wave
                            input_freq = input_wave.get('freq')
                            freq_idx = np.abs(wbf_freq - input_freq).argmin()
                            wbf_wave['amp'] = wbf_freq_signal_mag[freq_idx]
                            wbf_wave['freq'] = wbf_freq[freq_idx]
                            wbf_wave['phase'] = wbf_freq_signal_phase[freq_idx]
                            wbf_wave['real'] = wbf_freq_signal_real[freq_idx]
                            wbf_wave['imag'] = wbf_freq_signal_imag[freq_idx]
                            wbf_waves.append(wbf_wave)
                        
                        wbf_dut_wave_params.append(wbf_waves)
                    
                    with open(wbf_dut_wave_file, 'wb') as file:
                        pickle.dump(wbf_dut_wave_params, file)
                    self.logger.info(f"Wideband filtered DUT wave parameter file saved to {wbf_dut_wave_file}")
                else:
                    self.logger.error(f"{wbf_dut_signal_file} does not exist")
        
        self.logger.info("Wideband filtered DUT wave parameter creation complete\n")
        
    
    def create_premultiply_set(self,
                               dictionary_path=None,
                               input_config_name=None,
                               DUT_config_name=None,
                               normalize=None,
                               apply_fft=None,
                               fft_shift=None,
                               overwrite=None):
        self.logger.info(f"Starting Premultiply Set Creation...")
        
        if input_config_name is not None:
            self.set_input_config_name(input_config_name)
        if DUT_config_name is not None:
            self.set_DUT_config_name(DUT_config_name)
            
        if normalize is None:
            normalize = self.premultiply_params.get('normalize', False)
            
        if apply_fft is None:
            apply_fft = self.premultiply_params.get('apply_fft', False)

        if overwrite is None:
            overwrite = self.premultiply_params.get('overwrite', False)

        if fft_shift is None:
            fft_shift = self.premultiply_params.get('fft_shift', False) 
            
        wideband_freq_modes = self.freq_modes.get('wideband', [])
        output_dir = self.directories.get('outputs', "Outputs")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")    
        if dictionary_path is None:
            dictionary_filename = self.filenames.get('dictionary',"dictionary.npy")
            dictionary_path = output_dir / dictionary_filename
        if not dictionary_path.exists():
            self.logger.error("Dictionary File Does Not Exist")
            raise ValueError("Dictionary File Does Not Exist")
        dictionary = np.load(dictionary_path)
            
        premultiply_dir = self.directories.get('premultiply', "Premultiply")
        premultiply_dir.mkdir(parents=True, exist_ok=True)
        premultiply_filename = self.filenames.get('freq_signals', "freq_signals.npz")
        
        scale_dict = self.outputset_params.get('scale_dict', 1.0)
        scaled_dictionary = scale_dict * dictionary

        cp = import_module("cupy")
        Scaled_Dictionary = cp.asarray(scaled_dictionary, dtype=cp.complex64)
        Pinv_Dict = cp.linalg.pinv(Scaled_Dictionary)
        
        for file_path in output_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                stem = file_path.name
                key_part = stem.split(input_time_signal_filename)[0]
                premultiply_file = premultiply_dir / f"{key_part}{premultiply_filename}"
                
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
                temp_arr = {}
                if wideband_freq_modes:
                    for mode in wideband_freq_modes:
                        arr = fft_encode_signals(premultiply_signals, mode,
                                                 apply_fft=apply_fft,
                                                 apply_fftshift=fft_shift,
                                                 normalize=normalize)
                        #unsure why this step is necessary
                        arr = arr * 2
                        fd, path_arr = tempfile.mkstemp(suffix=".npy")
                        os.close(fd)
                        np.save(path_arr, arr)
                        temp_arr[mode] = path_arr                    

                    with ZipFile(premultiply_file, 'w', ZIP_DEFLATED) as zf:
                        for name, path in temp_arr.items():
                            zf.write(path, arcname=f"{name}.npy")
                    self.logger.info(f"{key_part[:-1]} wideband filter frequency set saved to {premultiply_file}")

                    for path in temp_arr.values():
                        os.remove(path)               
      
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
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        
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
        saved_freq_modes = self.freq_modes.get('recovery', [])
        
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
            ml_model_filename = self.filenames.get('ml_model', "ml_model.keras")
            
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
            
            
    def set_recovery_dataframe(self, saved_freq_modes=None):
        import re

        def numeric_key(s):
            # Extract the first number in the string
            m = re.search(r'\d+', s)
            return int(m.group()) if m else float('inf')
        
        def get_prefix_before_recovery(filename: str) -> str:
            lower = filename.lower()
            idx = lower.find("recovery")
            return filename[:idx] if idx != -1 else filename
        
        input_dir = self.directories.get('inputs', "Inputs")
        inputset_config_filename = self.flat_filenames.get('input.config', "inputset_config.json")
        input_time_signal_filename = self.flat_filenames.get('input.time_signal', "time_signals.npy")
        input_wave_params_filename = self.flat_filenames.get('input.wave_params', "wave_params.pkl")
        inputset_config_file = input_dir.parent / inputset_config_filename
        
        output_dir = self.directories.get('outputs', "Outputs")
        wbf_freq_filename = self.filenames.get('wbf_freq', "wbf_freq.npy")
        wbf_freq_file = output_dir / wbf_freq_filename
        wbf_freq = np.load(wbf_freq_file)

        DUT_config_filename = self.filenames.get('DUT_config', "DUT_config.json")
        DUT_config_file = output_dir.parent / DUT_config_filename
        
        recovery_dir = self.directories.get('recovery', "Recovery")
        recovery_config_filename = self.filenames.get('recovery_config', "recovery_config.json")
        recovery_config_file = recovery_dir.parent / recovery_config_filename

        recovery_df_filename = self.dataframe_params.get('file_path', "recovery_df.pkl")
        recovery_df_file_path = recovery_dir / recovery_df_filename
        recovery_df = pd.read_pickle(recovery_df_file_path)

        if not recovery_df_file_path.exists():
            self.logger.error(f"{recovery_df_file_path} does not exist.")
            raise ValueError(f"{recovery_df_file_path} does not exist.")
        else:
            recovery_df = pd.read_pickle(recovery_df_file_path)

        if not recovery_config_file.exists():
            self.logger.error(f"{recovery_config_file} does not exist")
            raise ValueError(f"{recovery_config_file} does not exist")
        else:
            recovery_config = load_config_from_json(recovery_config_file)
        
        recovery_config_name = recovery_config.get('config_name')        

        if not inputset_config_file.exists():
            self.logger.error(f"{inputset_config_file} does not exist")
            raise ValueError(f"{inputset_config_file} does not exist")
        else:
            inputset_config = load_config_from_json(inputset_config_file)
        
        inputset_config_name = inputset_config.get('config_name')
        inputset_params = inputset_config.get('inputset')
        num_recovery_sigs = inputset_params.get('num_recovery_sigs')

        if not DUT_config_file.exists():
            self.logger.error(f"{DUT_config_file} does not exist")
            raise ValueError(f"{DUT_config_file} does not exist")
        else:
            DUT_config = load_config_from_json(DUT_config_file)
        
        DUT_config_name = DUT_config.get('config_name')
        
        recovery_mag_threshold = self.dataframe_params.get('recovery_mag_thresh', 0.5)
        
        if saved_freq_modes is None:
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

            # npz modes must use .npz naming
            if mode in ("mag_ang_sincos", "mag_ang" ,"real_imag"):
                filename = str(Path(filename).with_suffix(".npz"))

            input_dict = {
                get_prefix_before_recovery(p.name): p
                for p in input_dir.iterdir()
                if p.is_file()
                and p.name.endswith(input_time_signal_filename)
                and "recovery" in p.name.lower()
            }

            recovery_dict = {
                get_prefix_before_recovery(p.name): p
                for p in recovery_dir.iterdir()
                if p.is_file()
                and p.name.endswith(filename)
                and "recovery" in p.name.lower()
            }

            wbf_wave_dict = {
                get_prefix_before_recovery(p.name): p
                for p in output_dir.iterdir()
                if p.is_file()
                and p.name.endswith(input_wave_params_filename)
                and "recovery" in p.name.lower()
            }

            wbf_freq_sig_dict = {
                get_prefix_before_recovery(p.name.replace("_wbf", "")): p
                for p in output_dir.iterdir()
                if p.is_file()
                and p.name.endswith(filename)
                and "recovery" in p.name.lower()
            }
            
            missing_in_wbf = recovery_dict.keys() - wbf_freq_sig_dict.keys()
            missing_in_recovery = wbf_freq_sig_dict.keys() - recovery_dict.keys()
            missing_in_input = recovery_dict.keys() - input_dict.keys()

            if missing_in_wbf:
                self.logger.error("Files missing in wideband filtered input:", missing_in_wbf)
                raise ValueError("Files missing in wideband filtered input:", missing_in_wbf)

            if missing_in_recovery:
                self.logger.error("Files missing in recovery:", missing_in_recovery)
                raise ValueError("Files missing in recovery:", missing_in_recovery)

            if missing_in_input:
                self.logger.error("Files missing in recovery:", missing_in_input)
                raise ValueError("Files missing in recovery:", missing_in_input)        
            
            sorted_keys = sorted(recovery_dict.keys(), key=numeric_key)

            matched_recovery_files = [recovery_dict[k] for k in sorted_keys]
            matched_wbf_files      = [wbf_freq_sig_dict[k] for k in sorted_keys]
            matched_wave_file      = [wbf_wave_dict[k] for k in sorted_keys]
            input_files            = [input_dict[k] for k in sorted_keys]
            
            for idx, recovery_file in enumerate(matched_recovery_files):
                input_file = input_files[idx]
                wbf_file = matched_wbf_files[idx]

                wbf_signals = np.load(wbf_file)
                recovered_signals = np.load(recovery_file)

                # -------------------------------
                # MODE: MAG + PHASE
                # -------------------------------
                if mode in ("mag_ang", "mag_ang_sincos"):
                    num_recovered_sigs = recovered_signals["complex_mag"].shape[0]
                    num_wbf_sigs = wbf_signals["complex_mag"].shape[0]
                
                # -------------------------------
                # MODE: REAL + IMAG
                # -------------------------------
                elif mode == "real_imag":
                    num_recovered_sigs = recovered_signals["real"].shape[0]
                    num_wbf_sigs = wbf_signals["real"].shape[0]

                else:
                    num_recovered_sigs = recovered_signals.shape[0]
                    num_wbf_sigs = wbf_signals.shape[0]
                
                wbf_wave_file = matched_wave_file[idx]
                with open(wbf_wave_file, "rb") as f:
                    wbf_waves = pickle.load(f)
                    
                if (num_recovered_sigs != num_recovery_sigs):
                    self.logger.error(f"Recovered signal set size {num_recovered_sigs} does not equal expected size {num_recovery_sigs}")
                    raise ValueError(f"Recovered signal set size {num_recovered_sigs} does not equal expected size {num_recovery_sigs}")
                if (num_wbf_sigs != num_recovery_sigs):
                    self.logger.error(f"Input signal set size {num_wbf_sigs} does not equal expected size {num_recovery_sigs}")
                    raise ValueError(f"Input signal set size {num_wbf_sigs} does not equal expected size {num_recovery_sigs}")  
                if (num_recovered_sigs != num_wbf_sigs):
                    self.logger.error(f"Input signal length {num_wbf_sigs} does not equal Recovered signal length {num_recovered_sigs}")
                    raise ValueError(f"Input signal length {num_wbf_sigs} does not equal Recovered signal length {num_recovered_sigs}")
                
                # Build the new row as a dict
                new_row = {
                    "input_file_name": input_file,
                    "wbf_file_name": wbf_file,
                    "recovery_file_name": recovery_file,
                    "input_config_name": inputset_config_name,
                    "DUT_config_name": DUT_config_name,
                    "recovery_config_name": recovery_config_name,
                    "Frequency_mode": mode,
                }

                # Currently only supporting magnitude stats
                for idx, rec_sig in enumerate(recovered_signals):
                    meta_data = self.__create_meta_data_dictionary(idx)

                    if mode in ("mag_ang", "mag_ang_sincos"):
                        rec_mag = rec_sig["complex_mag"]
                        rec_phase = rec_sig["complex_phase"]

                    wbf_wave = wbf_waves[idx]

                    # Extract freqs, amps, and phases
                    freqs = np.array([d['freq'] for d in wbf_wave])
                    amps = np.array([d['amp'] for d in wbf_wave]) / 2
                    phases = np.array([d['phase'] for d in wbf_wave])

                    # Positive and negative indices
                    pos_indices = np.array([np.argmin(np.abs(wbf_freq - f)) for f in freqs])
                    neg_indices = np.array([np.argmin(np.abs(wbf_freq + f)) for f in freqs])
                    wbf_unsorted_indices = np.concatenate([neg_indices, pos_indices])

                    # Combined amps and phases
                    amps_combined = np.concatenate([amps, amps])
                    phases_combined = np.concatenate([-phases, phases])

                    # Recovered tones
                    rec_sig_tones = np.where(rec_mag > recovery_mag_threshold)[0]
                    mask_recovered = np.isin(wbf_unsorted_indices, rec_sig_tones)

                    recovered_indices = rec_sig_tones[mask_recovered]
                    spur_indices = wbf_unsorted_indices[~mask_recovered]

                    rec_mag = np.abs(rec_mag[recovered_indices])
                    spur_mag = np.abs(rec_mag[spur_indices])

                    # Populate meta_data using dict comprehension
                    meta_data.update({
                        k: {'col_name': v['col_name'], 'value': val}
                        for k, v, val in [
                            ('num_rec_freq', meta_data['num_rec_freq'], recovered_indices.size),
                            ('num_spur_freq', meta_data['num_spur_freq'], spur_indices.size),
                            ('total_input_tones', meta_data['total_input_tones'], len(wbf_wave)),
                            ('rec_tone_thresh', meta_data['rec_tone_thresh'], recovery_mag_threshold),
                            ('ave_rec_mag_err', meta_data['ave_rec_mag_err'], abs(np.mean(amps_combined) - np.mean(rec_mag)) if rec_mag.size else -1),
                            ('ave_rec_mag', meta_data['ave_rec_mag'], np.mean(rec_mag) if rec_mag.size else -1),
                            ('max_rec_mag', meta_data['max_rec_mag'], np.max(rec_mag) if rec_mag.size else -1),
                            ('min_rec_mag', meta_data['min_rec_mag'], np.min(rec_mag) if rec_mag.size else -1),
                            ('ave_spur_mag', meta_data['ave_spur_mag'], np.mean(spur_mag) if spur_mag.size else -1),
                            ('max_spur_mag', meta_data['max_spur_mag'], np.max(spur_mag) if spur_mag.size else -1),
                            ('min_spur_mag', meta_data['min_spur_mag'], np.min(spur_mag) if spur_mag.size else -1)
                        ]
                    })

                    # Append meta_data to new_row
                    new_row.update({v['col_name']: v['value'] for v in meta_data.values()})
                
                # Append row
                recovery_df.loc[len(recovery_df)] = new_row

                
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
    
    
    def get_freq_modes(self):
        return self.freq_modes

    
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