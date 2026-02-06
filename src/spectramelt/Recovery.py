import numpy as np
from importlib import import_module
from .utils import(
    load_config_from_json,
    get_logger,
    compute_recovery_stats,
    flatten_dict,
    create_meta_data_dictionary,
    safe_min,
    safe_max,
    safe_mean,
    filter_valid_names,
    snr_db,
    enob_from_snr,
    VALID_SAVED_FREQ_MODES
)
from typing import Optional
from .protocols import MLPProtocol
import pandas as pd
import pickle
from pathlib import Path
import copy


class Recovery:
    VALID_RECOVERY_METHODS = {
        "iht", "omp", "spgl1", "mlp"
    }    
    VALID_RECOVERY_MODE_ANALYSIS = {
        "mag",
        "real_imag"
    }
    def __init__(self,
                all_params=None,
                freq_modes=None,
                recovery_params=None,
                dataframe_params=None,
                log_params=None,
                config_name=None,
                num_waves=1,
                config_file_path=None) -> None:
        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "freq_modes": freq_modes,
                "recovery_params": recovery_params,
                "dataframe_params": dataframe_params,
                "log_params": log_params,
                "config_name": config_name
            }
        
        self.set_all_params(all_params)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")

    # -------------------------------
    # Setters
    # -------------------------------

    def set_all_params(self, all_params=None):
        if all_params is None:
            all_params = {}
        
        freq_modes = all_params.get('freq_modes', None)
        recovery_params = all_params.get('recovery_params', None)
        dataframe_params = all_params.get('dataframe_params', None)
        log_params = all_params.get('log_params', None)
        premultiply_params = all_params.get('premultiply_params', None)
        
        config_name = all_params.get('config_name', "Recovery_Config_1")
        if recovery_params is None:
            config_name = "Default_Recovery_Config"

        self.set_log_params(log_params)           
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
            
        self.set_freq_modes(freq_modes)
        self.set_dataframe_params(dataframe_params)
        self.set_recovery_params(recovery_params)
        self.set_config_name(config_name)


    def set_freq_modes(self, freq_modes=None):
        if freq_modes is None:
            freq_modes = [
                "mag",
                "real_imag"
            ]

        valid_modes, removed_modes = filter_valid_names(freq_modes)
        if removed_modes:
            self.logger.warning(f"Invalid modes removed from frequency mode list: {removed_modes}")
        self.freq_modes = valid_modes
        
        
    def set_dataframe_params(self, dataframe_params=None):
        if dataframe_params is None:
            dataframe_params= {
                "file_path": "recovery_df.pkl",
                "save_as_csv": True,
                "recovery_mag_thresh": 0.1,
                "meta_column_names": {
                    "input_time_file_name": "str",
                    "wideband_filtered_file_name": "str",
                    "recovery_file_name": "str",
                    "dataset_config_name": "str",
                    "input_config_name": "str",
                    "DUT_config_name": "str",
                    "recovery_config_name": "str",
                    "Frequency_mode": "str",
                    "total_input_tones": "float64",
                    "rec_tone_thresh": "float64"
                },
                "signal_column_names": {
                    "num_rec_freq_" : "float64",
                    "num_spur_freq_": "float64",
                    "ave_rec_mag_err_": "float64",
                    "ave_rec_mag_": "float64",
                    "max_rec_mag_": "float64",
                    "min_rec_mag_": "float64",
                    "ave_spur_mag_": "float64",
                    "max_spur_mag_": "float64",
                    "min_spur_mag_": "float64"
                },
                "signal_column_stats": {
                    "ave_num_rec" : "float64",
                    "recovery_rate": "float64",
                    "ave_num_spur": "float64",
                    "ave_rec_mag_err": "float64",
                    "ave_rec_mag": "float64",
                    "max_rec_mag": "float64",
                    "min_rec_mag": "float64",
                    "ave_spur_mag": "float64",
                    "max_spur_mag": "float64",
                    "min_spur_mag": "float64"
                }
            }
        
        self.dataframe_params = dataframe_params


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
                "sigma": 0.001,
                "dict_mag_adj": 1.0
            }
        self.recovery_params = recovery_params
        self.set_recovery_type(recovery_params.get('recovery_type', None))
        self.set_recovery_method(recovery_params.get('method', None))
        
        
    def set_recovery_type(self, recovery_type: str):
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
        if self.is_valid_recovery_method(recovery_method):
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


    def is_valid_recovery_method(self, name) -> bool:
        if isinstance(name, str):
            name = [name]
        return all(n.lower() in self.VALID_RECOVERY_METHODS for n in name)


    def recover_signal(self, signal, dictionary=None, num_waves=1,
                       mlp: Optional[MLPProtocol] = None,
                       mlp_model=None, model_file_path=None,
                       recovery_type=None, recovery_method=None) -> np.ndarray:
        sigma = self.recovery_params.get('sigma', 0.001)
        dict_mag_adj = self.recovery_params.get('dict_mag_adj', 1.0)

        if recovery_method is None:
            recovery_method = self.recovery_params.get('method', None).lower()

        if recovery_type is None:
            recovery_type = self.recovery_params.get('recovery_type', "complex").lower()

        complex_recovery = {"complex", "imag", "ang", "mag_ang", "real_imag", "mag_ang_sincos"}
        # Assumes signals used to recover using the MLP are already premultiplied by the dictionary
        if recovery_method is None:
            self.logger.error("No recovery method specified")
            raise ValueError("No recovery method specified")
        elif recovery_method != "mlp":
            if dictionary is None:
                self.logger.error("No dictionary given")
                raise ValueError("No dictionary given")
            elif recovery_type in complex_recovery:
                if not np.iscomplexobj(dictionary):
                    self.logger.error("Dictionary is not complex")
                    raise ValueError("Dictionary is not complex")

        premultiply = self.recovery_params.get('premultiply', False)

        recovered_coef = None

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
            case 'mlp':
                if mlp is None:
                    self.logger.error("No MLP object given")
                    raise FileNotFoundError("No MLP object given")
                elif model_file_path is not None:
                    if not model_file_path.exists():
                        self.logger.error(f"File not found: {model_file_path}")
                        raise FileNotFoundError(f"File not found: {model_file_path}")
                    mlp.set_model_file_path(model_file_path)
                    
                if premultiply:
                    pseudo = np.linalg.pinv(dict_mag_adj *dictionary)
                    init_guess = np.dot(pseudo,signal)
                else:
                    init_guess = signal
                    
                recovered_coef = mlp.model_prediction(init_guess, recovery_type, mlp_model=mlp_model)
            case _:
                self.logger.error(f"Recovery method {recovery_method} is not supported")

        return recovered_coef


    def create_rec_df(self,
                      inputset_config_file: Path | None,
                      recovery_df_file_path: Path | None):
        inputset_config = load_config_from_json(inputset_config_file)
                        
        meta_column_names = self.dataframe_params.get('meta_column_names')
        signal_column_names = self.dataframe_params.get('signal_column_names')
        signal_column_stats = self.dataframe_params.get('signal_column_stats')
        input_config = inputset_config.get('inputset')
        num_recovery_sigs = input_config.get('num_recovery_sigs')
        
        # Build the master column dictionary
        full_column_dict = dict(meta_column_names)   # start with static columns

        for sig in range(num_recovery_sigs):
            for prefix, dtype in signal_column_names.items():
                full_column_dict[f"{prefix}{sig}"] = dtype

        for prefix, dtype in signal_column_stats.items():
            full_column_dict[f"{prefix}"] = dtype

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
            
            
    def process_signal_file(
        self,
        recovery_file: Path,
        rec_time_file: Path,
        wbf_file: Path,
        wbf_wave_file: Path,
        input_file: Path,
        num_recovery_sigs: int,
        dataset_config_name: str,
        inputset_config_name: str,
        DUT_config_name: str,
        recovery_config_name: str,
        wbf_freq: np.ndarray
    ) -> list[dict]:
        freq_modes = self.freq_modes
        dataframe_params = self.dataframe_params
        recovery_mag_threshold = dataframe_params.get('recovery_mag_thresh', 0.1)
        
        #Need to add support for real and imag modes in the future
        unsupported = set(freq_modes) - set(self.VALID_RECOVERY_MODE_ANALYSIS)

        if unsupported:
            self.logger.warning(f"Unsupported frequency modes {unsupported} found. Removing")
            freq_modes = [
                m for m in freq_modes
                if m not in unsupported
                ]

        # ---------- Load recovery ----------
        with np.load(recovery_file) as recovery_npz:
            available = set(recovery_npz.files)
            valid = [m for m in freq_modes if m in available]
            missing = set(freq_modes) - set(valid)
            if missing:
                print(f"Warning: Missing recovery modes in {recovery_file}: {missing}")
            recovery = {m: recovery_npz[m] for m in valid}
        
        # Time domain signals for SNR calculation
        rec_time_signals = np.load(rec_time_file)
        wbf_time_signals = np.load(wbf_file)

        # Split real_imag if present
        if "real_imag" in recovery:
            arr = recovery["real_imag"]
            real, imag = np.array_split(arr, 2, axis=1)
            recovery["real_imag"] = {"real": real, "imag": imag}

        # FFT shift
        for mode, data in recovery.items():
            if isinstance(data, dict):
                for k in data:
                    data[k] = np.fft.fftshift(data[k], axes=-1)
            else:
                recovery[mode] = np.fft.fftshift(data, axes=-1)

        flat_recovery = flatten_dict(recovery)

        # ---------- Sanity check ----------
        for k, arr in flat_recovery.items():
            if arr.shape[0] != num_recovery_sigs:
                raise ValueError(f"Recovered signal size mismatch: {k}")

        # ---------- Load WBF waves ----------
        with open(wbf_wave_file, "rb") as f:
            wbf_waves = pickle.load(f)

        rows = []

        # ================================================================
        # ONE ROW PER FREQUENCY MODE
        # ================================================================
        for mode in freq_modes:

            row = {
                "input_file_name": input_file,
                "wbf_file_name": wbf_wave_file,
                "recovery_file_name": recovery_file,
                "dataset_config_name": dataset_config_name,
                "input_config_name": inputset_config_name,
                "DUT_config_name": DUT_config_name,
                "recovery_config_name": recovery_config_name,
                "Frequency_mode": mode,
            }

            # ============================================================
            # PER-SIGNAL STATS
            # ============================================================
            for idx_sig, wbf_wave in enumerate(wbf_waves):

                if idx_sig == 0:
                    row["total_input_tones"] = len(wbf_wave)
                    row["rec_tone_thresh"] = recovery_mag_threshold

                # ---- reference data ----
                amps  = np.array([w["amp"]  for w in wbf_wave])
                freqs = np.array([w["freq"] for w in wbf_wave])
                reals = np.array([w["real"] for w in wbf_wave])
                imags = np.array([w["imag"] for w in wbf_wave])

                pos_idx = np.array([np.argmin(np.abs(wbf_freq - f)) for f in freqs])
                neg_idx = np.array([np.argmin(np.abs(wbf_freq + f)) for f in freqs])
                rec_bins = np.concatenate([neg_idx, pos_idx])

                all_bins = np.arange(wbf_freq.size)
                non_rec_bins = np.setdiff1d(all_bins, rec_bins)
                
                snr = snr_db(wbf_time_signals[idx_sig],
                             rec_time_signals[idx_sig])

                # ========================================================
                # MAG MODE
                # ========================================================
                if mode == "mag":
                    mag = flat_recovery["mag"][idx_sig]
                    mag_abs = np.abs(mag)

                    rec_vals = mag_abs[rec_bins]
                    rec_final = rec_vals[rec_vals > recovery_mag_threshold]

                    spur_vals = mag_abs[non_rec_bins]
                    spur_final = spur_vals[spur_vals > recovery_mag_threshold]

                    ref_vals = np.concatenate([amps, amps])

                # ========================================================
                # REAL / IMAG MODE
                # ========================================================
                elif mode == "real_imag":
                    real = flat_recovery["real_imag.real"][idx_sig]
                    imag = flat_recovery["real_imag.imag"][idx_sig]

                    real_abs = np.abs(real)
                    imag_abs = np.abs(imag)

                    # recovered (expected bins)
                    rec_real = real_abs[rec_bins]
                    rec_imag = imag_abs[rec_bins]

                    rec_final = np.concatenate([
                        rec_real[rec_real > recovery_mag_threshold],
                        rec_imag[rec_imag > recovery_mag_threshold],
                    ])

                    # spurs (unexpected bins)
                    spur_real = real_abs[non_rec_bins]
                    spur_imag = imag_abs[non_rec_bins]

                    spur_final = np.concatenate([
                        spur_real[spur_real > recovery_mag_threshold],
                        spur_imag[spur_imag > recovery_mag_threshold],
                    ])

                    ref_vals = np.concatenate([np.abs(reals), np.abs(imags),
                                            np.abs(reals), np.abs(imags)])

                else:
                    raise ValueError(f"Unsupported frequency mode: {mode}")

                # ---- compute stats ----
                stats = compute_recovery_stats(
                    rec_final,
                    spur_final,
                    ref_vals,
                    min_threshold=recovery_mag_threshold
                )

                # ---- write per-signal columns ----
                meta = create_meta_data_dictionary(idx_sig)
                meta["snr_db"]["value"] = snr
                meta["enob"]["value"] = enob_from_snr(snr)
                meta["rms_util"]["value"] = (
                    np.sqrt(np.mean(wbf_time_signals[idx_sig]**2)) / 10.0
                )
                (
                    meta["num_rec_freq"]["value"],
                    meta["num_spur_freq"]["value"],
                    meta["ave_rec_mag_err"]["value"],
                    meta["ave_rec_mag"]["value"],
                    meta["max_rec_mag"]["value"],
                    meta["min_rec_mag"]["value"],
                    meta["ave_spur_mag"]["value"],
                    meta["max_spur_mag"]["value"],
                    meta["min_spur_mag"]["value"],
                ) = stats

                row.update({v["col_name"]: v["value"] for v in meta.values()})

            # ============================================================
            # AGGREGATES
            # ============================================================
            num_rec_vals  = [row[f"num_rec_freq_{i}"] for i in range(num_recovery_sigs)]
            num_spur_vals = [row[f"num_spur_freq_{i}"] for i in range(num_recovery_sigs)]

            row["ave_num_rec"]  = safe_mean(num_rec_vals)
            row["ave_num_spur"] = safe_mean(num_spur_vals)

            denom = 2 if mode == "mag" else 4
            row["recovery_rate"] = (
                row["ave_num_rec"] / (denom * row["total_input_tones"])
                if row["ave_num_rec"] != -1 else -1
            )

            row["ave_rms_util"]    = safe_mean([row[f"rms_util_{i}"] for i in range(num_recovery_sigs)])
            row["ave_snr_db"]      = safe_mean([row[f"snr_db_{i}"] for i in range(num_recovery_sigs)])
            row["ave_enob"]        = safe_mean([row[f"enob_{i}"] for i in range(num_recovery_sigs)])
            row["ave_rec_mag_err"] = safe_mean([row[f"ave_rec_mag_err_{i}"] for i in range(num_recovery_sigs)])
            row["ave_rec_mag"]     = safe_mean([row[f"ave_rec_mag_{i}"] for i in range(num_recovery_sigs)])
            row["max_rec_mag"]     = safe_max([row[f"max_rec_mag_{i}"] for i in range(num_recovery_sigs)])
            row["min_rec_mag"]     = safe_min([row[f"min_rec_mag_{i}"] for i in range(num_recovery_sigs)])
            row["ave_spur_mag"]    = safe_mean([row[f"ave_spur_mag_{i}"] for i in range(num_recovery_sigs)])
            row["max_spur_mag"]    = safe_max([row[f"max_spur_mag_{i}"] for i in range(num_recovery_sigs)])
            row["min_spur_mag"]    = safe_min([row[f"min_spur_mag_{i}"] for i in range(num_recovery_sigs)])

            rows.append(row)

        return rows

    # -------------------------------
    # Getters
    # -------------------------------

    def get_config_name(self):
        return self.config_name
    
    
    def get_freq_modes(self):
        return self.freq_modes.copy()
    
    
    def get_dataframe_params(self):
        return copy.deepcopy(self.dataframe_params)

    
    def get_recovery_params(self):
        return self.recovery_params.copy()


    def get_log_params(self):
        return self.log_params.copy()
    
    
    def get_premultiply_params(self):
        return self.premultiply_params.copy()
    
    
    def get_all_params(self):
        all_params ={
            "config_name": self.config_name,
            "freq_modes": self.freq_modes,
            "recovery_params": self.recovery_params,
            "dataframe_params": self.dataframe_params,
            "log_params": self.log_params
        }
        return copy.deepcopy(all_params)