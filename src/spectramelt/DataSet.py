from .utils import (
    load_config_from_json,
    get_logger,
    build_flat_paths,
    flatten_files,
    find_project_root,
    save_to_json,
    fft_encode_signals,
    update_npz,
    get_prefix_before_recovery,
    numeric_key,
    REQUIRED_AXIS_KEYS
)
from .protocols import (
    InputSignalProtocol,
    AnalogProtocol,
    AllInputSetSignals,
    SignalSet,
    DUTProtocol,
    WaveParams,
    RecoveryProtocol,
    MLPProtocol
)
from types import ModuleType
from typing import (
    cast,
    Dict,
    Any   
)
import numpy as np
import pickle
import time
import copy
from importlib import import_module
from scipy.fft import fft, fftshift
import pandas as pd
from pathlib import Path
import platform
import tempfile
import os
from zipfile import ZipFile, ZIP_DEFLATED


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
            dataset_params = {
                "filenames": filenames,
                "directory_params": directory_params,
                "config_name": dataset_config_name,
                "log_params": log_params,
                "dataframe_params": dataframe_params,
                "seed": seed
            }
            
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
        filenames = dataset_params.get('filenames', None)
        directory_params = dataset_params.get('directory_params', None)
        log_params = dataset_params.get('log_params', None)
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
        # directory_params.setdefault('dataset_dir', "Data_Set")
        directory_params['dataset_dir'] = self.config_name
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

    # -------------------------------
    # Core functional methods
    # -------------------------------   
        
    def create_input_set(self,
                         analog: AnalogProtocol,
                         input_signal: InputSignalProtocol,
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
        def stringify_directories(dataset_params: dict) -> None:
            dirs = dataset_params.get("directories", {})
            for key, value in dirs.items():
                if isinstance(value, Path):
                    dirs[key] = str(value)

        self.logger.info("Starting Input Set Creation...")
        
        # --- Setup and pre-saves ---
        if input_signal is None:
            self.logger.error("Input Signal Object not set")
            raise ValueError("Input Signal Object Not Set")
        else:
            self.set_input_config_name(input_signal.get_config_name())
            
        inputset_params = input_signal.get_inputset_params()

        if normalize is None:
            normalize = inputset_params.get('normalize', False)

        if overwrite is None:
            overwrite = inputset_params.get('overwrite', False)

        if fft_shift is None:
            fft_shift = inputset_params.get('fft_shift', False)   
            
        input_dirs: Path = self.directories.get('inputs', "Inputs")
        real_time_freq_filename = self.filenames.get("real_time_freq", "real_time_freq.npz")
        dataset_config_filename = self.filenames.get('dataset_config', "dataset_config.json")
        inputset_config_filename = self.filenames.get('input_config', "inputset_config.json")
        input_wave_params_filename = self.filenames.get('wave_params', "wave_params.pkl")
        input_time_signals_filename = self.filenames.get('time_signals', "time_signals.npy")
        input_freq_signals_filename = self.filenames.get('freq_signals', "freq_signals.npz")
        
        input_freq_modes = input_signal.get_freq_modes()

        input_dirs.mkdir(parents=True, exist_ok=True)
        real_time_freq_file = input_dirs / real_time_freq_filename
        
        analog_signal = analog.create_analog()
        real_time = analog_signal.time
        real_freq = analog_signal.frequency
        if not real_time_freq_file.exists() or overwrite:
            np.savez(real_time_freq_file,
                     time=real_time,
                     freq=real_freq)
            self.logger.info(f"Real Time and Frequency Signal saved to file {real_time_freq_file}")

        # --- Dataset Config file ---
        dataset_config_file = input_dirs.parent.parent / dataset_config_filename
        dataset_params = self.get_all_params()
        stringify_directories(dataset_params)
        
        if not dataset_config_file.exists() or overwrite:
            save_to_json(dataset_params, dataset_config_file)
            self.logger.info(f"Saved Input Set configuration to file {dataset_config_file}")
        
        # --- Input Config file ---
        inputset_config_file = input_dirs.parent / inputset_config_filename
        input_signal_params = input_signal.get_all_params()
        input_signal_wave_params = input_signal_params.get('wave_params', None)
        if input_signal_wave_params is None:
            self.logger.error("Input Signal Wave Parameters not set")
            raise ValueError("Input Signal Wave Parameters Not Set")
        
        if not inputset_config_file.exists() or overwrite:
            inputset_config = {
                "config_name": self.config_name,
                "inputset": inputset_params,
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

        num_input_sigs = inputset_params.get('num_sigs', 1000)
        num_recovery_sigs = inputset_params.get('num_recovery_sigs', 100)
        tones_per_sig = inputset_params.get('tones_per_sig', [1])
        wave_precision = inputset_params.get('wave_precision', None)

        # --- Generate all tone sets ---
        for tones in tones_per_sig:
            all_inputset_signals: AllInputSetSignals = {
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
                input_signal_out = input_signal.create_input_signal(real_time)

                
                if input_sig == num_input_sigs:
                    inputset_type = "recovery"
                
                all_inputset_signals[inputset_type]["wave_set"].append(wave)
                all_inputset_signals[inputset_type]["time_set"].append(input_signal_out.input_signal)

            stop = time.time()
            self.logger.info(f"{num_input_sigs} {tones}-Tone Signal Input Set Creation Time: {stop - start:.6f} seconds")

            # --- Save outputs ---
            for set_info in all_inputset_signals.values():
                set_info = cast(SignalSet, set_info)
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


    def create_output_set(self, DUT: DUTProtocol,
                          analog: AnalogProtocol | None = None,
                          input_signal: InputSignalProtocol | None = None,
                          normalize=None,
                          fft_shift=None,
                          normalize_wbf=None,
                          fft_shift_wbf=None,
                          overwrite=None):
        self.logger.info(f"Starting Output Set Creation...")
        
        outputset_params = DUT.get_outputset_params()
        
        # --- Setup and pre-saves ---
        if input_signal is not None:
            self.set_input_config_name(input_signal.get_config_name())
            
        if normalize is None:
            normalize = outputset_params.get('normalize', False)

        if fft_shift is None:
            fft_shift = outputset_params.get('fft_shift', False)
            
        if normalize_wbf is None:
            normalize_wbf = outputset_params.get('normalize_wbf', False)

        if fft_shift_wbf is None:
            fft_shift_wbf = outputset_params.get('fft_shift_wbf', False)
            
        if overwrite is None:
            overwrite = outputset_params.get('overwrite', False)

        if DUT is None:
            self.logger.error("DUT Object not set")
            raise ValueError("DUT Object Not Set")
        else:
            self.set_DUT_config_name(DUT.get_config_name())
        
        input_dir: Path = self.directories.get('inputs', "Inputs")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        freq_signals_filename = self.filenames.get('freq_signals', "freq_signals.npz")
        
        real_time_freq_filename = self.filenames.get('real_time_freq', "real_time_freq.npz")
        real_time_freq_file = input_dir / real_time_freq_filename
        if real_time_freq_file.exists():
            real_time_freq = np.load(real_time_freq_file)
            real_time = real_time_freq["time"]
        elif analog is not None:
            real_time = analog.create_analog().time
        else:
            self.logger.error("No time file found and Input Signal Object not set")
            raise ValueError("No time file found and Input Signal Object Not Set")
        
        output_dirs: Path = self.directories.get('outputs', "Outputs")
        output_dirs.mkdir(parents=True, exist_ok=True)
        wideband_dir: Path = self.directories.get('wideband', "Wideband")
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
            
        freq_modes = DUT.get_freq_modes()
        output_freq_modes = freq_modes.get('output', [])
        wideband_freq_modes = freq_modes.get('wideband', [])
        
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
                    DUT_output_signals = DUT.create_output_signal(signal, real_time,
                                                                  return_wbf=True)
                    quantized_signals = DUT_output_signals.adc_signal.quantized
                    wbf_signal = DUT_output_signals.wbf_signal.wbf_sub_sig
                    output_signal = quantized_signals.quantized_values
                    output_signal_list.append(output_signal)
                    wbf_signal_list.append(wbf_signal)

                    if idx == 0:
                        if not dictionary_file.exists() or overwrite:
                            match DUT_type.lower():
                                case "nyfr":               
                                    lo_phase_mod_mid = DUT_output_signals.lo_phase_mod_mid
                                    wbf_time = DUT_output_signals.wbf_signal.time
                                    dictionary = DUT.create_dictionary(lo_phase_mod_mid, wbf_time)
                            np.save(dictionary_file, dictionary)
                            self.logger.info(f"DUT {DUT_type} Dictionary saved to file {dictionary_file}")
                            
                        if not samp_time_freq_file.exists() or overwrite:
                            np.savez(samp_time_freq_file,
                                     time=quantized_signals.mid_times,
                                     freq=quantized_signals.sampled_frequency)
                            self.logger.info(f"DUT {DUT_type} sample time and frequency array saved to file {samp_time_freq_file}")
                            
                        if not wbf_time_freq_file.exists() or overwrite:
                            np.savez(wbf_time_freq_file,
                                     time=DUT_output_signals.wbf_signal.time,
                                     freq=DUT_output_signals.wbf_signal.freq)
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
        input_dir: Path = self.directories.get('inputs', "Inputs")
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
            

    def create_nyfr_wave_params(self, nyfr: DUTProtocol):
        self.logger.info(f"Starting NYFR folded wave parameter Creation...")
        
        outputset_params = nyfr.get_outputset_params()
        DUT_type = outputset_params.get('DUT_type', None).lower()
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
        input_dir: Path = self.directories.get('inputs', "Inputs")
        input_wave_params_filename = self.filenames.get('wave_params', "wave_params.pkl")
        output_dir: Path = self.directories.get('outputs', "Outputs")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        samp_time_freq_filename = self.filenames.get('samp_time_freq', "sampled_time_freq.npz")
        samp_time_freq = np.load(output_dir / samp_time_freq_filename)
        samp_freq: np.ndarray = samp_time_freq["freq"]
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
                            input_wave = cast(WaveParams, input_wave)
                            nyfr_wave = input_wave
                            input_freq = input_wave.get('freq')
                            folded_freq: np.ndarray = np.abs(input_freq - LO_freq * round(input_freq/LO_freq))
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

        input_dir: Path = self.directories.get('inputs', "Inputs")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        input_wave_params_filename = self.filenames.get('wave_params', "wave_params.pkl")
        wideband_dir: Path = self.directories.get('wideband', "Wideband")

        wbf_time_freq_filename = self.filenames.get('wbf_time_freq', "wbf_time_freq.npz")      
        wbf_time_freq_file = wideband_dir / wbf_time_freq_filename
        
        if not wbf_time_freq_file.exists():
            self.logger.error(f"{wbf_time_freq_file} does not exist")
            raise ValueError(f"{wbf_time_freq_file} does not exist")
        wbf_time_freq = np.load(wbf_time_freq_file)
        wbf_freq: np.ndarray = wbf_time_freq["freq"]
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
                            input_wave = cast(WaveParams, input_wave)
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
        
    
    def create_premultiply_set(self, DUT: DUTProtocol,
                               recovery: RecoveryProtocol | None,
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
            
        premultiply_params = {}
        if recovery is not None:
            premultiply_params = recovery.get_premultiply_params()
            
        if normalize is None:
            normalize = premultiply_params.get('normalize', False)
            
        if apply_fft is None:
            apply_fft = premultiply_params.get('apply_fft', False)

        if overwrite is None:
            overwrite = premultiply_params.get('overwrite', False)

        if fft_shift is None:
            fft_shift = premultiply_params.get('fft_shift', False) 
        
        freq_modes = DUT.get_freq_modes()
        wideband_freq_modes = freq_modes.get('wideband', [])
        output_dir: Path = self.directories.get('outputs', "Outputs")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")    
        if dictionary_path is None:
            dictionary_filename = self.filenames.get('dictionary',"dictionary.npy")
            dictionary_path = output_dir / dictionary_filename
        if not dictionary_path.exists():
            self.logger.error("Dictionary File Does Not Exist")
            raise ValueError("Dictionary File Does Not Exist")
        dictionary = np.load(dictionary_path)
            
        premultiply_dir: Path = self.directories.get('premultiply', "Premultiply")
        premultiply_dir.mkdir(parents=True, exist_ok=True)
        premultiply_filename = self.filenames.get('freq_signals', "freq_signals.npz")
        
        scale_dict = premultiply_params.get('scale_dict', 1.0)
        scaled_dictionary = scale_dict * dictionary


        cp: ModuleType = import_module("cupy")
            
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
                            recovery: RecoveryProtocol,
                            mlp: MLPProtocol | None,
                            dictionary_path=None,
                            input_config_name=None,
                            DUT_config_name=None):
        self.logger.info(f"Starting Recovery Set Creation...")
        
        output_dir: Path = self.directories.get('outputs', "Outputs")
        premultiply_dir: Path = self.directories.get('premultiply', "Premultiply")
        wideband_dir: Path = self.directories.get('wideband', "Wideband")    
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        freq_signals_filename = self.filenames.get('freq_signals', "freq_signals.npz")
        
        if input_config_name is not None:
            self.set_input_config_name(input_config_name)
        if DUT_config_name is not None:
            self.set_DUT_config_name(DUT_config_name)
            
        if dictionary_path is None:
            output_dirs: Path = self.directories.get('outputs', "Outputs")
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
    
        recovery_dirs: Path = self.directories.get('recovery', "Recovery")
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
        freq_modes = recovery.get_freq_modes()
        
        if recovery_method != "mlp":
            for file_path in output_dir.iterdir():
                if (file_path.is_file() and 
                    file_path.name.endswith(input_time_signal_filename) and 
                    "recovery" in file_path.name.lower()):
                    
                    output_signals = np.load(file_path)
                    recovered_sig_list = []

                    self.logger.info(f"Starting Recovery Set Creation for {file_path}")
                    start = time.time() 
                    
                    for signal in output_signals:
                            recovered_sig_list.append(recovery.recover_signal(signal, dictionary))
                    stop = time.time()
                    self.logger.info(f"{len(output_signals)} Signal Recovery Set Creation Time: {stop - start:.6f} seconds")
                            
                    np.save(recovery_file, np.array(recovered_sig_list))
                    self.logger.info(f"Recovery Set Creation Complete for Output Set {file_path} using Recovery Method {recovery_method}")
        elif freq_modes:
            ml_models_dir: Path = self.directories.get('ml_models', "ML_Models")
            ml_model_filename = self.filenames.get('ml_model', "ml_model.keras")

            for mode in freq_modes:
                if mlp is None:
                    self.logger.error("No MLP object given")
                    raise ValueError("No MLP object given")
                
                training_params = mlp.get_training_params()
                norm_params = training_params.get('norm_params', None)
                if norm_params is not None:
                    input_norm_type = norm_params.get('input_type', None)
                    output_norm_type = norm_params.get('output_type', None)

                    if input_norm_type is not None:
                        norm_premultiply_h5_file = premultiply_dir / f"{Path(freq_signals_filename).stem}_{mode}_{input_norm_type}.h5"
                        if not norm_premultiply_h5_file.exists():
                            self.logger.warning(f"{norm_premultiply_h5_file} file does not exist")
                        else:
                            mlp.set_recovery_stats_from_h5(norm_premultiply_h5_file, dataset_name="X")
                    else:
                        self.logger.info("No normalization type specified for input set")
                    
                    if output_norm_type is not None:
                        norm_output_h5_file = wideband_dir / f"wbf_{Path(freq_signals_filename).stem}_{mode}_{output_norm_type}.h5"
                        if not norm_output_h5_file.exists():
                            self.logger.warning(f"{norm_output_h5_file} file does not exist")
                        else:
                            mlp.set_recovery_stats_from_h5(norm_output_h5_file, dataset_name="y")
                    else:
                        self.logger.info("No normalization type specified for output set")
                
                else:
                    self.logger.warning("No normalization parameters found for the MLP")
                
                ml_model_file = ml_models_dir / f"{mode}_{ml_model_filename}"
                if not ml_model_file.exists():
                    self.logger.error(f"{ml_model_file} file does not exist")
                    raise ValueError(f"{ml_model_file} file does not exist")
                mlp.set_model_file_path(ml_model_file)
                mlp_model = mlp.load_model()
                
                recovery.set_recovery_type(mode)

                for file_path in premultiply_dir.iterdir():
                    if (file_path.is_file() and 
                        file_path.name.endswith(freq_signals_filename) and 
                        "recovery" in file_path.name.lower()):

                        freq_dict = {}
                        all_output_signals = np.load(file_path)
                        # Extract identifying portion
                        stem = file_path.name
                        key_part = stem.split(freq_signals_filename)[0]
                        recovery_file = recovery_dirs / f"{key_part}{freq_signals_filename}"
                    

                        output_signals = all_output_signals[mode]        
                        recovered_sig_list = []

                        self.logger.info(f"Starting Recovery Set Creation for {file_path}")
                        start = time.time() 
                        
                        for signal in output_signals:
                                recovered_sig_list.append(recovery.recover_signal(signal, MLP=mlp, mlp_model=mlp_model))

                        stop = time.time()
                        self.logger.info(f"{len(output_signals)} Signal Recovery Set Creation Time: {stop - start:.6f} seconds")
                        
                        freq_dict[mode] = np.array(recovered_sig_list)
                        update_npz(recovery_file, **freq_dict)
                        self.logger.info(f"Recovery set saved to {file_path} for recovery method {recovery_method}")
                
                mlp.reset_tensorflow_session()
        else:
            self.logger.error(f"Recovery method is {recovery_method} but no save frequency list given")
            raise ValueError(f"Recovery method is {recovery_method} but no save frequency list given")
        
        self.logger.info("Recovery Set Creation Complete\n")


    def decode_time_signals(self):
        """
        Decode all frequency-domain .npz files in the recovery directory
        into time-domain signals and save them as .npy files.

        Priority for reconstruction:
            1. 'real_imag'  -> concatenated [real0, imag0, real1, imag1, ...]
            2. 'mag_ang'    -> concatenated [mag0, ang0, mag1, ang1, ...]
            3. 'mag_ang_sincos' -> concatenated [mag0, sin0, cos0, mag1, sin1, cos1, ...]

        Signals are assumed Physics-normalized (denormalize by multiplying by signal length).
        """
        recovery_dir = Path(self.directories.get('recovery', "Recovery"))
        time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        freq_signals_filename = self.filenames.get('freq_signals', "freq_signals.npz")

        recovery_dir.mkdir(parents=True, exist_ok=True)

        for file_path in recovery_dir.iterdir():
            if not file_path.is_file() or not file_path.name.endswith(freq_signals_filename):
                continue

            # Key for saving
            stem = file_path.stem
            key_part = stem.split(freq_signals_filename)[0]
            recovery_time_file = recovery_dir / f"{key_part}{time_signal_filename}"

            with np.load(file_path) as freq_data:
                keys = freq_data.files
                time_signal = None

                # Priority 1: real_imag
                if 'real_imag' in keys:
                    arr: np.ndarray = freq_data['real_imag']
                    arr = arr.reshape(-1, 2)  # shape (N, 2)
                    real = arr[:, 0]
                    imag = arr[:, 1]
                    time_signal = np.fft.ifft(real + 1j*imag) * arr.shape[0]

                # Priority 2: mag_ang
                elif 'mag_ang' in keys:
                    arr: np.ndarray = freq_data['mag_ang']
                    arr = arr.reshape(-1, 2)  # shape (N, 2)
                    mag = arr[:, 0]
                    ang = arr[:, 1]
                    complex_freq = mag * np.exp(1j * ang)
                    time_signal = np.fft.ifft(complex_freq) * arr.shape[0]

                # Priority 3: mag_ang_sincos
                elif 'mag_ang_sincos' in keys:
                    arr: np.ndarray = freq_data['mag_ang_sincos']
                    arr = arr.reshape(-1, 3)  # shape (N, 3)
                    mag = arr[:, 0]
                    sin_comp = arr[:, 1]
                    cos_comp = arr[:, 2]
                    ang = np.arctan2(sin_comp, cos_comp)
                    complex_freq = mag * np.exp(1j * ang)
                    time_signal = np.fft.ifft(complex_freq) * arr.shape[0]

                else:
                    self.logger.warning(f"Cannot reconstruct time signal from {file_path}")
                    continue

                # Save the time-domain signal
                np.save(recovery_time_file, time_signal)
                self.logger.info(f"Saved time-domain signal: {recovery_time_file}")

        
    def create_recovery_dataframe(self, recovery: RecoveryProtocol | None):
        if recovery is None:
            self.logger.error("No recovery object provided")
            raise ValueError("No recovery object provided")
        
        self.logger.info("Creating Dataframe for recovery signals")
        # --- Config file ---
        input_dirs: Path = self.directories.get('inputs', "Inputs")
        inputset_config_filename = self.flat_filenames.get('input.config', "inputset_config.json")
        inputset_config_file: Path = input_dirs.parent / inputset_config_filename
        
        recovery_dirs: Path = self.directories.get('recovery', "Recovery")
        dataframe_params = recovery.get_dataframe_params()
        recovery_df_filename = dataframe_params.get('file_path', "recovery_df.pkl")
        recovery_df_file_path = recovery_dirs / recovery_df_filename

        if recovery_df_file_path.exists():
            self.logger.warning(f"{recovery_df_file_path} exists.  Will be overwritten")
            
        if not inputset_config_file.exists():
            self.logger.error(f"{inputset_config_file} does not exist")
            raise ValueError(f"{inputset_config_file} does not exist")
        
        recovery.create_rec_df(inputset_config_file,
                               recovery_df_file_path)
            
            
    def set_recovery_dataframe(self, recovery: RecoveryProtocol | None):
        if recovery is None:
            self.logger.error("No recovery object provided")
            raise ValueError("No recovery object provided")

        dataframe_params = recovery.get_dataframe_params()
           
        input_dir: Path = self.directories.get('inputs', "Inputs")
        inputset_config_filename = self.filenames.get('input_config', "inputset_config.json")
        input_time_signal_filename = self.filenames.get('time_signals', "time_signals.npy")
        freq_signals_filename = self.filenames.get('freq_signals', "freq_signals.npz",)
        wave_params_filename = self.flat_filenames.get('wave_params', "wave_params.pkl")
        inputset_config_file = input_dir.parent / inputset_config_filename
        
        wideband_dir: Path = self.directories.get('wideband', "Wideband")
        wbf_time_freq_filename = self.filenames.get('wbf_time_freq', "wbf_time_freq.npz")
        time_freq_file = wideband_dir / wbf_time_freq_filename
        with np.load(time_freq_file) as time_freq:
            missing = [k for k in REQUIRED_AXIS_KEYS if k not in time_freq]
            if missing:
                raise ValueError(f"{time_freq_file} missing required arrays: {missing}")
            wbf_freq = time_freq["freq"]

        # --- Dataset Config file ---
        dataset_config_filename = self.filenames.get('dataset_config', "dataset_config.json")
        dataset_config_file = input_dir.parent.parent / dataset_config_filename

        output_dir: Path = self.directories.get('outputs', "Outputs")
        DUT_config_filename = self.filenames.get('DUT_config', "DUT_config.json")
        DUT_config_file = output_dir.parent / DUT_config_filename
        
        recovery_dir: Path = self.directories.get('recovery', "Recovery")
        recovery_config_filename = self.filenames.get('recovery_config', "recovery_config.json")
        recovery_config_file = recovery_dir.parent / recovery_config_filename

        recovery_df_filename = dataframe_params.get('file_path', "recovery_df.pkl")
        recovery_df_file_path = recovery_dir / recovery_df_filename

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
        
        recovery_recovery: Dict[str, Any] = recovery_config.get('recovery')
        recovery_config_name = recovery_recovery.get('config_name')

        if not dataset_config_file.exists():
            self.logger.error(f"{dataset_config_file} does not exist")
            raise ValueError(f"{dataset_config_file} does not exist")
        else:
            dataset_config = load_config_from_json(dataset_config_file)

        dataset_config_name = dataset_config.get('config_name')   

        if not inputset_config_file.exists():
            self.logger.error(f"{inputset_config_file} does not exist")
            raise ValueError(f"{inputset_config_file} does not exist")
        else:
            inputset_config = load_config_from_json(inputset_config_file)
        inputset_input: Dict[str, Any] = inputset_config.get('input')
        inputset_config_name = inputset_input.get('config_name')
        inputset_params: Dict[str, Any] = inputset_config.get('inputset')
        num_recovery_sigs: int = inputset_params.get('num_recovery_sigs')

        if not DUT_config_file.exists():
            self.logger.error(f"{DUT_config_file} does not exist")
            raise ValueError(f"{DUT_config_file} does not exist")
        else:
            DUT_config = load_config_from_json(DUT_config_file)
        
        DUT_output = DUT_config.get('output')
        DUT_config_name = DUT_output.get('config_name')

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
            and p.name.endswith(freq_signals_filename)
            and "recovery" in p.name.lower()
        }

        wbf_wave_dict = {
            get_prefix_before_recovery(p.name): p
            for p in wideband_dir.iterdir()
            if p.is_file()
            and p.name.endswith(wave_params_filename)
            and "recovery" in p.name.lower()
        }
        
        missing_in_wbf = recovery_dict.keys() - wbf_wave_dict.keys()
        missing_in_recovery = wbf_wave_dict.keys() - recovery_dict.keys()
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
        matched_wave_files     = [wbf_wave_dict[k] for k in sorted_keys]
        input_files            = [input_dict[k] for k in sorted_keys]

        all_rows = []

        for idx_file, recovery_file in enumerate(matched_recovery_files):
            rows = recovery.process_signal_file(
                recovery_file=recovery_file,
                wbf_wave_file=matched_wave_files[idx_file],
                input_file=input_files[idx_file],
                num_recovery_sigs=num_recovery_sigs,
                dataset_config_name=dataset_config_name,
                inputset_config_name=inputset_config_name,
                DUT_config_name=DUT_config_name,
                recovery_config_name=recovery_config_name,
                wbf_freq=wbf_freq
            )
            all_rows.extend(rows)

        signal_column_stats = dataframe_params.get('signal_column_stats')

        # --- Build final dataframe ---
        recovery_df = pd.DataFrame(all_rows)
        recovery_df.to_pickle(recovery_df_file_path)

        significant_digits = 4
        # Round each column to 4 significant digits
        for col in signal_column_stats:
            if col in recovery_df.columns:
                # Use np.format_float_positional to maintain significant digits, then convert back to float
                recovery_df[col] = recovery_df[col].apply(
                    lambda x: float(np.format_float_positional(x, precision=significant_digits, unique=False, trim='k')) 
                    if pd.notnull(x) else x
                )

        if dataframe_params.get('save_as_csv', True):
            recovery_df.to_csv(recovery_df_file_path.with_suffix(".csv"), index=False)

        
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
        return self.directories.copy()


    def get_flat_filenames(self):
        return self.flat_filenames.copy()
    
    
    def get_filenames(self):
        return self.filenames.copy()


    def get_log_params(self):
        return self.log_params.copy()
    

    def get_directory_params(self):
        return copy.deepcopy(self.directory_params)
    
    
    def get_valid_dut_types(cls):
        return cls.VALID_DUT_TYPES.copy()
    
    
    def get_all_params(self):
        all_params = {
            "config_name": self.config_name,
            "DUT_config_name": self.DUT_config_name,
            "input_config_name": self.input_config_name,
            "recovery_config_name": self.recovery_config_name,
            "ML_config_name": self.ML_config_name,
            "directory_params": self.directory_params,
            "log_params": self.log_params,
            "directories": self.directories,
            "filenames": self.filenames,
        }
        return copy.deepcopy(all_params)