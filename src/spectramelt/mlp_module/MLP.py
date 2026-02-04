import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
from spectramelt.utils import (
    load_config_from_json,
    get_logger,
    filter_valid_names
)
import tensorflow as tf
import h5py
import time
from keras import (
    layers,
    losses,
    backend,
    Sequential,
    optimizers,
    Input,
    Model
)
from keras.callbacks import EarlyStopping
from keras.activations import get as get_activation
from sklearn.model_selection import train_test_split
from .losses import (
    resolve_loss,
)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class MLP:
    """
    """
    # Sets for fast membership checking
    CUSTOM_LOSSES = {"root_mean_squared_error",
                     "weighted_root_mean_squared_error",
                     "weighted_rmse_with_energy",
                     "hubersparseamplitudeloss"}
    VALID_MONITORS = {
        "loss", "val_loss", "accuracy",
        "acc", "val_accuracy", "val_acc"
    }
    VALID_NORM_TYPES = {
        "zscore", "maxabs", "minmax"
    }
    VALID_NORM_SCOPE = {
        "global",
        "elementwise"
    }
    
    def __init__(self,
                 mlp_params=None,
                 freq_modes=None,
                 model_params=None,
                 premultiply_params=None,
                 training_params=None,
                 log_params=None,
                 config_name=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------

        """
        if config_file_path is not None:
            mlp_params = load_config_from_json(config_file_path)
        elif mlp_params is None:
            mlp_params = {}
            mlp_params['model_params'] = model_params
            mlp_params['freq_modes'] = freq_modes
            mlp_params['premultiply_params'] = premultiply_params
            mlp_params['training_params'] = training_params
            mlp_params['config_name'] = config_name
            mlp_params['log_params'] = log_params

        self.set_mlp_params(mlp_params)

        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")

 
    # -------------------------------
    # Setters
    # -------------------------------
        
    def set_mlp_params(self, mlp_params=None):
        if mlp_params is None:
            mlp_params = {}
        model_params = mlp_params.get('model_params', None)
        premultiply_params = mlp_params.get('premultiply_params', None)
        freq_modes = mlp_params.get('freq_modes', None)
        training_params = mlp_params.get('training_params', None)
        config_name = mlp_params.get('config_name', None)
        log_params = mlp_params.get('log_params', None)
        
        if ( mlp_params is None and
            log_params is None):
            config_name = "Default_MLP_Config"
        
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
            
        self.set_freq_modes(freq_modes)
        self.set_premultiply_params(premultiply_params)
        self.set_model_params(model_params)
        self.set_training_params(training_params)
        self.set_config_name(config_name)
        
    
    def set_config_name(self, config_name):
        if config_name is None:
            config_name = "MLP_Config_1"
        self.config_name = config_name
        
    
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
                        
    def set_premultiplty_params(self, premultiply_params=None):
        if premultiply_params is None:
            premultiply_params = {
                "scale_dict": 0.5,
                "normalize": True,
                "apply_fft": False,
                "fft_shift": False,
                "overwrite": True
            }
            
        self.premultiply_params = premultiply_params
        

    def set_model_params(self, model_params):
        if model_params is None:
            model_params = {
                "file_path": "ml_model.keras",
                "hidden_layer_width_mult": [1.0],
                "activation_per_layer": [
                    "linear",
                    "linear"
                ]
            }
        activation_per_layer = model_params.get('activation_per_layer',
                                                ["linear", "linear"])
        if not self.validate_activation_list(activation_per_layer):
            self.logger.error("Activation function list contains invalid entries")
            raise ValueError("Activation function list contains invalid entries")
        self.model_params = model_params
        
    
    def set_log_params(self, log_params):
        if log_params is None:
            log_params = {
                "enabled": True,
                "log_file": None,
                "level": "INFO",
                "console": True
            }
        self.log_params = log_params
        
    
    def set_training_params(self, training_params):
        if training_params is None:
            training_params = {
                "total_num_sigs": 40000,
                "test_fraction": 0.3,
                "norm_params": {
                    "input_type": "maxabs",
                    "input_scope": "global",
                    "output_type": "maxabs",
                    "output_scope": "global"
                },
                "seed": None,
                "shuffle": True,
                "loss_type": "HuberSparseAmplitudeLoss",
                "loss_params": {
                    "tau": 0.02,
                    "amplitude_weight": 0.1,
                    "delta": 1.0
                },
                "learning_rate": 0.00005,
                "num_epochs": 100,
                "batch_sz": 256,
                "early_stopping": {
                    "monitor": "val_loss",
                    "min_delta": 0.005,
                    "patience": 25,
                    "start_from_epoch": 10,
                    "restore_best_weights": True
                }
            }
            
        self.rng = np.random.default_rng(training_params.get('seed', None))
        
        loss_type = training_params.get('loss_type', None)
        loss_type = loss_type.lower().strip()
        if not self.is_valid_keras_loss(loss_type):
            self.logger.error(f"{loss_type} is not a valid Keras loss function")
            raise ValueError(f"{loss_type} is not a valid Keras loss function")
        else:
            training_params['loss_type'] = loss_type

        test_fraction = training_params.get('test_fraction', None)
        if not self.is_valid_test_fraction(test_fraction):
            self.logger.error(f"{test_fraction} is not a valid percentage")
            raise ValueError(f"{test_fraction} is not a valid percentage")
        
        norm_params = training_params.get('norm_params', None)

        if norm_params is not None:
            input_type = norm_params.get('input_type', None)
            output_type = norm_params.get('output_type', None)

            if input_type is not None:
                if not self.is_valid_norm_type(input_type):
                    self.logger.error(f"{input_type} is not a valid input norm type")
                    raise ValueError(f"{input_type} is not a valid input norm type")
            
            if output_type is not None:
                if not self.is_valid_norm_type(output_type):
                    self.logger.error(f"{output_type} is not a valid output norm type")
                    raise ValueError(f"{output_type} is not a valid output norm type")
            
            input_scope = norm_params.get('input_scope', None)
            output_scope = norm_params.get('output_scope', None)

            if input_scope is not None:
                if not self.is_valid_norm_scope(input_scope):
                    self.logger.error(f"{input_scope} is not a valid input norm type")
                    raise ValueError(f"{input_scope} is not a valid input norm type")
            
            if output_scope is not None:
                if not self.is_valid_norm_scope(output_scope):
                    self.logger.error(f"{output_scope} is not a valid output norm type")
                    raise ValueError(f"{output_scope} is not a valid output norm type")
                
            self.set_input_recovery_stats(norm_type=input_type, norm_scope=input_scope)
            self.set_output_recovery_stats(norm_type=output_type, norm_scope=output_scope) 
  
        early_stopping_params = training_params.get('early_stopping', {})
        monitor = early_stopping_params.get('monitor', "val_loss")
        if not self.is_valid_monitor(monitor):
            self.logger.error(f"{monitor} is not a valid Keras monitor value")
            raise ValueError(f"{monitor} is not a valid Keras monitor value")        
        
        self.training_params = training_params


    def set_model_file_path(self, model_file_path):
        if model_file_path is None:
            model_file_path = "ml_model.keras"
        if self.model_params is None:
            self.logger.error("Model parameters not set")
            raise ValueError("Model parameters not set")
        
        self.model_params["file_path"] = model_file_path


    def set_recovery_stats_from_h5(self, norm_h5_path: Path, dataset_name: str):
        if norm_h5_path is None:
            raise ValueError("h5 file path cannot be None")

        if not norm_h5_path.is_file():
            raise ValueError(f"{norm_h5_path} is not a valid file")

        # ------------------------------------------------------------
        # Check that the file name contains a valid normalization type
        # ------------------------------------------------------------
        filename_lower = norm_h5_path.name.lower()
        if not any(norm_type in filename_lower for norm_type in self.VALID_NORM_TYPES):
            raise ValueError(
                f"{norm_h5_path} does not appear to contain a normalized dataset "
                f"(expected one of: {sorted(self.VALID_NORM_TYPES)})"
        )

        # ------------------------------------------------------------
        # Load normalization metadata
        # ------------------------------------------------------------
        with h5py.File(norm_h5_path, "r") as f:
            if "normalization" not in f:
                raise ValueError("Normalization group missing in HDF5 file")

            norm_grp = f["normalization"]

            norm_type = norm_grp.attrs.get("method")
            if norm_type is None:
                raise ValueError("Normalization type missing in HDF5 metadata")
            
            norm_scope = norm_grp.attrs.get("scope")
            if norm_scope is None:
                raise ValueError("Normalization scope missing in HDF5 metadata")
            
            # Load optional fields safely
            mean  = norm_grp["mean"][:]  if "mean"  in norm_grp else None
            scale = norm_grp["scale"][:] if "scale" in norm_grp else None
            min_  = norm_grp["min"][:]   if "min"   in norm_grp else None
            max_  = norm_grp["max"][:]   if "max"   in norm_grp else None

        # ------------------------------------------------------------
        # Validate normalization type and scope
        # ------------------------------------------------------------
        if not self.is_valid_norm_type(norm_type):
            raise ValueError(f"Unsupported normalization type: {norm_type}")
        
        if not self.is_valid_norm_scope(norm_scope):
            raise ValueError(f"Unsupported normalization scope: {norm_scope}")

        # ------------------------------------------------------------
        # Dispatch to appropriate setter
        # ------------------------------------------------------------
        if dataset_name == "X":
            self.set_input_recovery_stats(
                mean=mean,
                scale=scale,
                norm_type=norm_type,
                norm_scope=norm_scope,
                min_val=min_,
                max_val=max_,
            )
        elif dataset_name == "y":
            self.set_output_recovery_stats(
                mean=mean,
                scale=scale,
                norm_type=norm_type,
                norm_scope=norm_scope,
                min_val=min_,
                max_val=max_,
            )
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        

    def set_input_recovery_stats(
        self,
        mean=None,
        scale=None,
        norm_type=None,
        norm_scope=None,
        min_val=None,
        max_val=None,
    ):
        self.input_norm_type = norm_type
        self.input_norm_scope = norm_scope
        self.input_mean = mean
        self.input_scale = scale
        self.input_min = min_val
        self.input_max = max_val
        # ---------------- Logging ----------------
        self.logger.info("=== Input Recovery Statistics Loaded ===")
        self.logger.info(f"Normalization type : {self.input_norm_type}")
        self.logger.info(f"Normalization type : {self.input_norm_scope}")

        if self.input_mean is not None:
            self.logger.info(f"Mean shape/value  : {getattr(self.input_mean, 'shape', 'scalar')} | {self.input_mean}")
        else:
            self.logger.info("Mean              : None")

        if self.input_scale is not None:
            self.logger.info(f"Scale shape/value : {getattr(self.input_scale, 'shape', 'scalar')} | {self.input_scale}")
        else:
            self.logger.info("Scale             : None")

        if self.input_min is not None:
            self.logger.info(f"Min value shape   : {getattr(self.input_min, 'shape', 'scalar')} | {self.input_min}")
        else:
            self.logger.info("Min value         : None")

        if self.input_max is not None:
            self.logger.info(f"Max value shape   : {getattr(self.input_max, 'shape', 'scalar')} | {self.input_max}")
        else:
            self.logger.info("Max value         : None")

        self.logger.info("=======================================")


    def set_output_recovery_stats(
        self,
        mean=None,
        scale=None,
        norm_type=None,
        norm_scope=None,
        min_val=None,
        max_val=None,
    ):
        self.output_norm_type = norm_type
        self.output_norm_scope = norm_scope
        self.output_mean = mean
        self.output_scale = scale
        self.output_min = min_val
        self.output_max = max_val

        # ---------------- Logging ----------------
        self.logger.info("=== Output Recovery Statistics Loaded ===")
        self.logger.info(f"Normalization type : {self.output_norm_type}")
        self.logger.info(f"Normalization type : {self.output_norm_scope}")

        if self.output_mean is not None:
            self.logger.info(f"Mean shape/value  : {getattr(self.output_mean, 'shape', 'scalar')} | {self.output_mean}")
        else:
            self.logger.info("Mean              : None")

        if self.output_scale is not None:
            self.logger.info(f"Scale shape/value : {getattr(self.output_scale, 'shape', 'scalar')} | {self.output_scale}")
        else:
            self.logger.info("Scale             : None")

        if self.output_min is not None:
            self.logger.info(f"Min value shape   : {getattr(self.output_min, 'shape', 'scalar')} | {self.output_min}")
        else:
            self.logger.info("Min value         : None")

        if self.output_max is not None:
            self.logger.info(f"Max value shape   : {getattr(self.output_max, 'shape', 'scalar')} | {self.output_max}")
        else:
            self.logger.info("Max value         : None")

        self.logger.info("========================================")

    # -------------------------------
    # Core functional methods
    # -------------------------------

    @classmethod
    def is_valid_norm_type(cls, norm_type) -> bool:
        if not isinstance(norm_type, str):
            return False
        return norm_type.lower() in cls.VALID_NORM_TYPES

    @classmethod
    def is_valid_norm_scope(cls, norm_scope) -> bool:
        if not isinstance(norm_scope, str):
            return False
        return norm_scope.lower() in cls.VALID_NORM_SCOPE

    def is_valid_keras_activation(self, name):
        """Check if a single activation name is valid in Keras."""
        try:
            get_activation(name)
            return True
        except (ValueError, TypeError):
            return False


    def validate_activation_list(self, activation_list):
        """Validate a list of activation names."""
        if not isinstance(activation_list, (list, tuple)):
            self.logger.error("activation_list must be a list or tuple of strings")
            raise TypeError("activation_list must be a list or tuple of strings")
        
        invalid = [a for a in activation_list if not self.is_valid_keras_activation(a)]
        if invalid:
            self.logger.info(f"Invalid activation(s): {invalid}")
            return False
        return True


    def is_valid_keras_loss(self, name: str) -> bool:
        # Check custom losses first
        if name in self.CUSTOM_LOSSES:
            return True
        
        # Check Keras built-ins
        try:
            losses.get(name)
            return True
        except ValueError:
            return False
        
    
    def is_valid_monitor(self, name: str) -> bool:
        return name.lower() in self.VALID_MONITORS
    
    
    def is_valid_test_fraction(self, value):
        """Return True if value is a float between 0 and 1 inclusive."""
        try:
            value = float(value)  # Convert in case it’s a string or int
        except (TypeError, ValueError):
            return False
        return 0 <= value <= 1   

    
    def reset_tensorflow_session(self):
        import gc

        backend.clear_session()     # Proper Keras reset
        tf.keras.backend.clear_session()
        gc.collect()          # Force Python GC
        # Force GPU memory release
        try:
            from numba import cuda
            cuda.select_device(0)
            cuda.close()
        except:
            pass


    def create_model(self, input_signal_size, output_signal_size, model_file_path=None):
        if model_file_path is None:
            model_file_path = self.model_params.get('file_path', "ml_model.keras")

        loss_type = self.training_params.get('loss_type', "mean_squared_error")
        loss_params = self.training_params.get('loss_params', {})
        learning_rate = self.training_params.get('learning_rate', 0.00001)

        hidden_layer_width_mult = self.model_params.get('hidden_layer_width_mult', [1])
        activation_functions = self.model_params.get(
            'activation_per_layer', ["linear", "linear"]
        )

        mlp_model = Sequential()
        mlp_model.add(Input(shape=(input_signal_size,), name="mlp_model_in"))

        for idx, layer_width_mult in enumerate(hidden_layer_width_mult):
            size = int(layer_width_mult * input_signal_size)
            activation_function = activation_functions[idx]
            layer_name = f"mlp_model_layer_{idx}"
            mlp_model.add(layers.Dense(size,
                                    activation=activation_function,
                                    name=layer_name))

        mlp_model.add(layers.Dense(output_signal_size,
                                activation=activation_functions[-1],
                                name="mlp_model_out"))

        mlp_opt = optimizers.Adam(learning_rate=learning_rate)

        loss_fn = resolve_loss(loss_type, loss_params)

        mlp_model.compile(
            optimizer=mlp_opt,
            loss=loss_fn
        )

        mlp_model.save(model_file_path, overwrite=True)

        return mlp_model
        

    def load_model(self, model_file_path: Path | None) -> Model:
        if model_file_path is None:
            model_file_path = self.model_params.get('file_path', "ml_model.keras")

        if not model_file_path.exists():
            self.logger.error(f"Model file {model_file_path} does not exist")
            raise ValueError(f"Model file {model_file_path} does not exist")

        mlp_model = tf.keras.models.load_model(model_file_path)

        return mlp_model   
    
        
    def fit_model(self, input_set, output_set, model_file_path=None):
        mlp_model = self.load_model(model_file_path)
        
        test_fraction = self.training_params.get('test_fraction', 0.3)
        # Using np.random.Generator
        random_state = self.rng.integers(low=0, high=2**32)
        
        shuffle = self.training_params.get('shuffle', False)
        num_epochs = self.training_params.get('num_epochs', 200)
        batch_sz = self.training_params.get('batch_sz', 128) 
        early_stopping_params = self.training_params.get('early_stopping', {})
        monitor = early_stopping_params.get('monitor', "val_loss")
        min_delta = early_stopping_params.get('min_delta', 0.1)
        patience = early_stopping_params.get('patience', 4)
        verbose = early_stopping_params.get('verbose', 1)
        start_from_epoch = early_stopping_params.get('start_from_epoch', 5)
        restore_best_weights = early_stopping_params.get('restore_best_weights', True)
        
        early_stopping = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience,
                                       verbose=verbose, start_from_epoch=start_from_epoch,
                                       restore_best_weights=restore_best_weights)
        
        X_train, X_test, y_train, y_test = train_test_split(
            input_set, output_set, test_fraction=test_fraction,
            random_state=random_state, shuffle=shuffle
        )
        
        mlp_model.fit(
            X_train, y_train, epochs=num_epochs, batch_size=batch_sz,
            shuffle=True, validation_data=(X_test, y_test), callbacks=[early_stopping]
        )
        
        mlp_model.save(model_file_path, overwrite=True)


    def model_prediction(self, init_guess, mode,
                         norm_h5_input_path=None,
                         norm_h5_output_path=None,
                         mlp_model=None,
                         in_norm_type=None,
                         out_norm_type=None) -> np.array:

        if (norm_h5_input_path is not None and
            norm_h5_input_path.is_file() and 
            any(tag in norm_h5_input_path.name.lower() for tag in self.VALID_NORM_TYPES)):
            self.set_recovery_stats_from_h5(norm_h5_input_path, dataset_name="X")

        if in_norm_type is None:
            in_norm_type = self.input_norm_type

        coef_predict = None
        if mlp_model is None:
            mlp_model = self.load_model()
        if ( mode == "complex" ):
            self.logger.error("Complex values not supported by MLPs")
            raise ValueError("Complex values not supported by MLPs")
        else:
            # ---------------- Normalize input signal ----------------
            if in_norm_type is not None:
                if in_norm_type == "zscore":
                    init_guess = (init_guess - self.input_mean) / self.input_scale

                elif in_norm_type == "maxabs":
                    init_guess = init_guess / self.input_scale

                elif in_norm_type == "minmax":
                    init_guess  = (init_guess  - self.input_min) / (self.input_max - self.input_min)
                
            reshaped_guess = init_guess.reshape((1, init_guess.shape[0]))
            reshaped_coef_predict = mlp_model.predict(reshaped_guess)
            coef_predict = reshaped_coef_predict.reshape(-1)

        if (norm_h5_output_path is not None and
            norm_h5_output_path.is_file() and 
            any(tag in norm_h5_output_path.name.lower() for tag in self.VALID_NORM_TYPES)):
            self.set_recovery_stats_from_h5(norm_h5_output_path, dataset_name="y")

        if out_norm_type is None:
            out_norm_type = self.output_norm_type

        # ---------------- Denormalize signal ----------------
        if out_norm_type is not None:
            if out_norm_type == "zscore":
                coef_predict = coef_predict * self.output_scale + self.output_mean

            elif out_norm_type == "maxabs":
                coef_predict = coef_predict * self.output_scale

            elif out_norm_type == "minmax":
                coef_predict = coef_predict * (self.output_max - self.output_min) + self.output_min
        
        return coef_predict
        
    # -------------------------------
    # New: large-dataset ingestion + training
    # -------------------------------

    def prepare_large_dataset(
            self,
            input_file_list,
            output_file_list,
            h5_out_input: str,
            h5_out_output: str,
            mode: str,
            sample_signal: np.ndarray,
            max_signals_per_file: int | None = None
    ):
        """
        Convert multiple .npz frequency mode files into two large HDF5 datasets (X and y),
        written sequentially for speed and then globally shuffled in-place.

        Parameters
        ----------
        input_file_list : list[str]
            List of input .npz files containing frequency modes.
        output_file_list : list[str]
            List of output .npz files containing frequency modes.
        h5_out_input : str
            Path to HDF5 input dataset.
        h5_out_output : str
            Path to HDF5 output dataset.
        mode : str
            Frequency mode to extract from each file ("mag", "ang", "real", "imag").
        sample_signal : np.ndarray
            Example signal for shape/dtype inference.
        max_signals_per_file : int | None
            Optional maximum number of signals per file to include.
        """

        # -------------------------------
        # 1. Determine chunk sizes
        # -------------------------------
        sample_signal = np.asarray(sample_signal)
        item_bytes = sample_signal.nbytes
        target_chunk_bytes = 1 * 1024 * 1024  # 1 MB
        batch = max(1, target_chunk_bytes // item_bytes)

        chunk_shape_in = (batch, *sample_signal.shape)
        if sample_signal.ndim == 1:
            out_shape = sample_signal.shape
            chunk_shape_out = (batch, sample_signal.shape[0])
        else:
            out_shape = sample_signal.shape[:-1]
            chunk_shape_out = (batch, *out_shape)
        input_dtype = sample_signal.dtype

        # -------------------------------
        # 2. Count total samples
        # -------------------------------
        total_samples = 0
        for in_file in input_file_list:
            X_block_npz = np.load(in_file, mmap_mode='r')
            if mode not in X_block_npz.files:
                raise ValueError(f"Mode '{mode}' not found in {in_file}")
            n = X_block_npz[mode].shape[0]
            if max_signals_per_file:
                n = min(n, max_signals_per_file)
            total_samples += n

        self.logger.info(f"Total samples across all files: {total_samples}")

        # -------------------------------
        # 3. Create HDF5 datasets
        # -------------------------------
        with h5py.File(h5_out_input, "w") as hf_in, \
            h5py.File(h5_out_output, "w") as hf_out:

            dset_in = hf_in.create_dataset(
                "X",
                shape=(total_samples, *sample_signal.shape),
                maxshape=(None,) + sample_signal.shape,
                dtype=input_dtype,
                chunks=chunk_shape_in,
            )

            dset_out = hf_out.create_dataset(
                "y",
                shape=(total_samples, *out_shape),
                maxshape=(None,) + out_shape,
                dtype="float32",
                chunks=chunk_shape_out,
            )

            # -------------------------------
            # 4. Sequential write
            # -------------------------------
            write_pos = 0
            for in_file, out_file in zip(input_file_list, output_file_list):

                X_block_npz = np.load(in_file, mmap_mode='r')
                y_block_npz = np.load(out_file, mmap_mode='r')

                if mode not in X_block_npz.files:
                    raise ValueError(f"Mode '{mode}' not found in {in_file}")
                if mode not in y_block_npz.files:
                    raise ValueError(f"Mode '{mode}' not found in {out_file}")

                X_block = X_block_npz[mode]
                y_block = y_block_npz[mode]

                n_block = X_block.shape[0]
                if max_signals_per_file:
                    n_block = min(n_block, max_signals_per_file)

                self.logger.debug(f"[WRITE] {in_file}  → indices [{write_pos}:{write_pos+n_block}]")

                dset_in[write_pos:write_pos + n_block] = X_block[:n_block]
                dset_out[write_pos:write_pos + n_block] = y_block[:n_block]

                write_pos += n_block

            hf_in.flush()
            hf_out.flush()

        # -------------------------------
        # 5. Deterministic global shuffle
        # -------------------------------
        self.logger.info("Applying deterministic global shuffle…")
        perm = self.rng.permutation(total_samples)

        with h5py.File(h5_out_input, "r+") as hf_in, \
            h5py.File(h5_out_output, "r+") as hf_out:

            X = hf_in["X"]
            y = hf_out["y"]

            Xtmp = hf_in.create_dataset("X_shuffled", shape=X.shape, dtype=X.dtype, chunks=X.chunks)
            ytmp = hf_out.create_dataset("y_shuffled", shape=y.shape, dtype=y.dtype, chunks=y.chunks)

            batch = 8192
            for start in range(0, total_samples, batch):
                end = min(start + batch, total_samples)
                idx = perm[start:end]
                sorted_idx = np.sort(idx)
                X_block_sorted = X[sorted_idx]
                y_block_sorted = y[sorted_idx]
                inv_order = np.argsort(np.argsort(idx))
                X_block = X_block_sorted[inv_order]
                y_block = y_block_sorted[inv_order]
                Xtmp[start:end] = X_block
                ytmp[start:end] = y_block

            del hf_in["X"]
            del hf_out["y"]
            hf_in.move("X_shuffled", "X")
            hf_out.move("y_shuffled", "y")

        self.logger.info("[DONE] Dataset creation and shuffle complete.")
       
        
    def scan_hdf5_stats(
        self,
        h5_path,
        dataset_name,
        batch_size=4096,
        norm_scope="global",  # "global" or "elementwise"
    ):
        with h5py.File(h5_path, "r") as f:
            data = f[dataset_name]
            N = data.shape[0]

            if norm_scope == "elementwise":
                feat_dim = int(np.prod(data.shape[1:]))
            else:
                feat_dim = 1

            # Initialize accumulators
            sum_x = np.zeros(feat_dim)
            sum_x2 = np.zeros(feat_dim)
            min_val = np.full(feat_dim, np.inf)
            max_val = np.full(feat_dim, -np.inf)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = data[start:end].astype(np.float64)

                if norm_scope == "elementwise":
                    batch = batch.reshape(batch.shape[0], -1)
                else:
                    batch = batch.reshape(-1)

                sum_x += batch.sum(axis=0)
                sum_x2 += np.square(batch).sum(axis=0)
                min_val = np.minimum(min_val, batch.min(axis=0))
                max_val = np.maximum(max_val, batch.max(axis=0))

            eps = 1e-8
            mean = sum_x / (N if norm_scope == "global" else (N * batch.shape[1]))
            var = (sum_x2 / (N if norm_scope == "global" else (N * batch.shape[1]))) - mean**2
            std = np.sqrt(np.maximum(var, eps))
            scale = np.maximum(max_val - min_val, eps)

            return {
                "scope": norm_scope,
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "scale": scale,
            }
    

    def normalize_hdf5_dataset(
        self,
        input_h5_path,
        dataset_name,
        batch_size=4096,
        output_dataset_name=None,
        dtype=np.float32,
        norm_type="zscore",
        norm_scope="global",
    ):
        input_h5_path = Path(input_h5_path)

        stats = self.scan_hdf5_stats(
            input_h5_path,
            dataset_name,
            batch_size=batch_size,
            norm_scope=norm_scope,
        )

        output_h5_path = input_h5_path.with_name(
            input_h5_path.stem + f"_{norm_type}" + input_h5_path.suffix
        )

        if output_dataset_name is None:
            output_dataset_name = dataset_name

        self.logger.info(f"Normalizing {input_h5_path} → {output_h5_path}")

        with h5py.File(input_h5_path, "r") as f_in, \
            h5py.File(output_h5_path, "w") as f_out:

            data = f_in[dataset_name]
            N = data.shape[0]

            out = f_out.create_dataset(
                output_dataset_name,
                shape=data.shape,
                dtype=dtype,
                chunks=True,
                compression="gzip",
            )

            # ---- Save metadata ----
            norm_grp = f_out.create_group("normalization")
            norm_grp.attrs["method"] = norm_type
            norm_grp.attrs["scope"] = norm_scope

            for k, v in stats.items():
                if isinstance(v, np.ndarray):
                    norm_grp.create_dataset(k, data=v)
                else:
                    norm_grp.attrs[k] = v

            # ---- Apply normalization ----
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = data[start:end].astype(np.float64)

                if norm_scope == "elementwise":
                    batch = batch.reshape(batch.shape[0], -1)

                if norm_type == "zscore":
                    batch = (batch - stats["mean"]) / stats["std"]
                elif norm_type == "maxabs":
                    batch = batch / stats["max"]
                elif norm_type == "minmax":
                    batch = (batch - stats["min"]) / stats["scale"]

                out[start:end] = batch.reshape((end - start,) + data.shape[1:])

                if start % (10 * batch_size) == 0:
                    self.logger.info(f"Normalized {start}/{N}")

        self.logger.info(f"Saved normalized dataset → {output_h5_path}")
        return output_h5_path


    def make_hdf5_batch_loader(self, h5_input_path, h5_output_path):
        """
        Returns a function that TensorFlow can call inside tf.py_function
        to load HDF5 slices efficiently. Handles arbitrary index order.
        """
        hf_x = h5py.File(h5_input_path, "r")
        hf_y = h5py.File(h5_output_path, "r")
        X = hf_x["X"]
        Y = hf_y["y"]
        N = X.shape[0]

        def get_batch(indices):
            # Convert Tensor -> numpy array
            indices_np = indices.numpy()

            # HDF5 fancy indexing requires sorted indices
            sorted_idx = np.sort(indices_np)
            rev = np.argsort(np.argsort(indices_np))  # to restore original order

            # Efficient contiguous reads
            batch_x_sorted = X[sorted_idx]
            batch_y_sorted = Y[sorted_idx]

            # Restore original index order
            batch_x = batch_x_sorted[rev].astype("float32")
            batch_y = batch_y_sorted[rev].astype("float32")

            return batch_x, batch_y

        return get_batch, N
    

    def train_on_hdf5(self, h5_input_path, h5_output_path, model_file_path=None, ):
        """
        Train the existing Keras model on large HDF5 datasets using efficient batched loading.
        Splits train/test deterministically before batching to avoid empty datasets.
        """
        if model_file_path is None:
            model_file_path = self.model_params.get('file_path', "ml_model.keras")
            if not model_file_path.is_file():
                self.logger.error(f"{model_file_path} does not exist")
                raise ValueError(f"{model_file_path} does not exist")

        if not h5_input_path.is_file():
            self.logger.error(f"{h5_input_path} does not exist")
            raise ValueError(f"{h5_input_path} does not exist")

        if not h5_output_path.is_file():
            self.logger.error(f"{h5_output_path} does not exist")
            raise ValueError(f"{h5_output_path} does not exist")

        mlp_model = self.load_model(model_file_path)

        num_epochs = self.training_params.get("num_epochs", 200)
        batch_sz   = self.training_params.get("batch_sz", 128)
        test_frac  = self.training_params.get("test_fraction", 0.3)

        # Early stopping
        early_stopping_params = self.training_params.get('early_stopping', {})
        early_stopping = EarlyStopping(
            monitor=early_stopping_params.get('monitor', "val_loss"),
            min_delta=early_stopping_params.get('min_delta', 0.1),
            patience=early_stopping_params.get('patience', 4),
            verbose=early_stopping_params.get('verbose', 1),
            start_from_epoch=early_stopping_params.get('start_from_epoch', 5),
            restore_best_weights=early_stopping_params.get('restore_best_weights', True),
        )

        norm_h5_input_path = h5_input_path
        if self.input_norm_type is not None:
            if not any(self.input_norm_type in h5_input_path.name.lower()
                       for self.input_norm_type in self.VALID_NORM_TYPES):
                    norm_h5_input_path = self.normalize_hdf5_dataset(h5_input_path,
                                                                    dataset_name="X",
                                                                    norm_type=self.input_norm_type,
                                                                    norm_scope=self.input_norm_scope)
                    self.set_recovery_stats_from_h5(norm_h5_input_path, dataset_name="X")

        norm_h5_output_path = h5_output_path
        if self.output_norm_type is not None:
            if not any(self.output_norm_type in h5_output_path.name.lower()
                    for self.output_norm_type in self.VALID_NORM_TYPES):
                    norm_h5_output_path = self.normalize_hdf5_dataset(h5_output_path,
                                                                    dataset_name="y",
                                                                    norm_type=self.output_norm_type,
                                                                    norm_scope=self.output_norm_scope)
                    self.set_recovery_stats_from_h5(norm_h5_output_path, dataset_name="y")

        # Build HDF5 loader
        get_batch, total_num_sigs = self.make_hdf5_batch_loader(norm_h5_input_path, norm_h5_output_path)

        # Generate train/test split BEFORE batching
        all_indices = np.arange(total_num_sigs)
        self.rng.shuffle(all_indices)  # deterministic if self.rng seeded

        test_size = int(total_num_sigs * test_frac)
        test_indices  = all_indices[:test_size]
        train_indices = all_indices[test_size:]

        # Convert indices to TF datasets
        train_ds = tf.data.Dataset.from_tensor_slices(train_indices)
        test_ds  = tf.data.Dataset.from_tensor_slices(test_indices)

        # Batch indices
        train_ds = train_ds.batch(batch_sz, drop_remainder=False)
        test_ds  = test_ds.batch(batch_sz, drop_remainder=False)

        # Map indices -> actual data
        train_ds = train_ds.map(
            lambda idx: tf.py_function(func=get_batch, inp=[idx], Tout=(tf.float32, tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        test_ds = test_ds.map(
            lambda idx: tf.py_function(func=get_batch, inp=[idx], Tout=(tf.float32, tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Prefetch to GPU
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

        self.logger.info(f"Starting mlp model training for {model_file_path}")
        start = time.time()
        # Train
        mlp_model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=num_epochs,
            callbacks=[early_stopping]
        )
        stop = time.time()
        self.logger.info(f"{total_num_sigs} Signal Training and Testing Time for {model_file_path}: {stop - start:.6f} seconds")
        mlp_model.save(model_file_path, overwrite=True)
        
    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_log_params(self):        
        return self.log_params
    
    
    def get_freq_modes(self):
        return self.freq_modes
    
    
    def get_premultiply_params(self):
        return self.premultiply_params
    
    
    def get_training_params(self):
        return self.training_params
    
    
    def get_model_params(self):
        return self.model_params
    
    
    def get_input_recovery_stats(self):
        input_recovery_stats = {
            "norm_type": self.input_norm_type,
            "norm_scope": self.input_norm_scope,
            "mean": self.input_mean,
            "scale": self.input_scale,
            "min_val": self.input_min,
            "max_val": self.input_max
        }
        return input_recovery_stats        
    
    
    def get_output_recovery_stats(self):
        output_recovery_stats = {
            "norm_type": self.output_norm_type,
            "norm_scope": self.output_norm_scope,
            "mean": self.output_mean,
            "scale": self.output_scale,
            "min_val": self.output_min,
            "max_val": self.output_max
        }
        return output_recovery_stats
    

    def get_recovery_stats(self):
        recovery_stats = {
            "input": self.get_input_recovery_stats(),
            "output": self.get_output_recovery_stats()
        }
        return recovery_stats

    
    def get_config_name(self):
        return self.config_name
        
    
    def get_all_params(self):
        all_params ={
            "log_params": self.log_params,
            "freq_modes": self.freq_modes,
            "premultiply_params": self.premultiply_params,
            "training_params": self.training_params,
            "model_params": self.model_params,
            "config_name": self.config_name
        }
        return all_params