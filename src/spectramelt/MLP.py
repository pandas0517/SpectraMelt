import numpy as np
from .utils import load_config_from_json, get_logger
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras import (
    layers,
    losses,
    backend,
    Sequential,
    Input,
    optimizers,
)
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split

class MLP:
    """
    """
    # Sets for fast membership checking
    CUSTOM_LOSSES = {"root_mean_squared_error"}
    VALID_MONITORS = {
        "loss", "val_loss", "accuracy",
        "acc", "val_accuracy", "val_acc"
    }
    
    def __init__(self,
                 mlp_params=None,
                 model_file_path=None,
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
            mlp_params['model_file_path'] = model_file_path
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
        model_file_path = mlp_params.get('model_file_path', None)
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
        
        self.set_model_file_path(model_file_path)
        self.set_training_params(training_params)
        self.set_config_name(config_name)
        
    
    def set_config_name(self, config_name):
        if config_name is None:
            config_name = "MLP_Config_1"
        self.config_name = config_name
        

    def set_model_file_path(self, model_file_path):
        if model_file_path is None:
            model_file_path = "ml_model.keras"
        self.model_file_path = model_file_path
        
    
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
                "modes": [
                    "mag"
                ],
                "total_num_sigs": 40000,
                "test_fraction": 0.3,
                "loss_type": "root_mean_squared_error",
                "learning_rate": 0.00001,
                "num_epochs": 200,
                "batch_sz": 128,
                "early_stopping": {
                    "monitor": "val_loss",
                    "min_delta": 0.1,
                    "patience": 4,
                    "verbose": 1,
                    "start_from_epoch": 5,
                    "restore_best_weights" :True
                },
                "seed": None,
                "shuffle": False
            }
            
        self.rng = np.random.default_rng(training_params.get('seed', None))
        
        loss_type = training_params.get('loss_type', None)
        loss_type = loss_type.lower().strip()
        if not self.is_valid_keras_loss(loss_type):
            self.logger.error(f"{loss_type} is not a valid Keras loss function")
            raise ValueError(f"{loss_type} is not a valid Keras loss function")
        else:
            training_params['loss_type'] = loss_type

        train_test_split = training_params.get('train_test_split', None)
        if not self.is_valid_train_test_split(train_test_split):
            self.logger.error(f"{train_test_split} is not a valid percentage")
            raise ValueError(f"{train_test_split} is not a valid percentage")
  
        early_stopping_params = training_params.get('early_stopping', {})
        monitor = early_stopping_params.get('monitor', "val_loss")
        if not self.is_valid_monitor(monitor):
            self.logger.error(f"{monitor} is not a valid Keras monitor value")
            raise ValueError(f"{monitor} is not a valid Keras monitor value")           
        
        self.training_params = training_params

    # -------------------------------
    # Core functional methods
    # -------------------------------
    
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
    
    
    def is_valid_train_test_split(self, split):
        """Return True if value is a float between 0 and 1 inclusive."""
        try:
            value = float(value)  # Convert in case it’s a string or int
        except (TypeError, ValueError):
            return False
        return 0 <= value <= 1   
    
    def root_mean_squared_error(self, y_true, y_pred):
        return tf.math.sqrt(losses.mean_squared_error(y_true, y_pred))

    
    def reset_tensorflow_session(self):
        backend.clear_session()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.keras.backend.clear_session()


    def create_model(self, input_signal_size, model_file_path=None):
        if model_file_path is None:
            model_file_path = self.model_file_path
            
        loss_type = self.training_params.get('loss_type', "mean_squared_error")
        learning_rate = self.training_params.get('learning_rate', 0.00001)

        
        mlp_model = Sequential()
        mlp_model.add(Input(shape=(input_signal_size,)))
        mlp_model.add(layers.Dense(input_signal_size, name="mlp_model_layer_1"))
        mlp_model.add(layers.Dense(input_signal_size, name="mlp_model_out"))
        mlp_opt = optimizers.Adam(learning_rate=learning_rate)
        if loss_type == "root_mean_squared_error":
            mlp_model.compile(optimizer=mlp_opt, loss=self.root_mean_squared_error)
        else:
            mlp_model.compile(optimizer=mlp_opt, loss=loss_type)
        
        mlp_model.save(model_file_path, overwrite=True)
        
    def fit_model(self, input_set, output_set, model_file_path=None):
        if model_file_path is None:
            model_file_path = self.model_file_path
            
        loss_type = self.training_params.get('loss_type', "mean_squared_error")
        if model_file_path.exists():
            if loss_type == "root_mean_squared_error":
                mlp_model = tf.keras.models.load_model(
                    model_file_path, custom_objects={'root_mean_squared_error': self.root_mean_squared_error}
                )
            else:
                mlp_model = tf.keras.models.load_model(model_file_path)
        else:
            self.logger.error(f"Model File {model_file_path} does not exists")
            raise ValueError(f"Model File {model_file_path} does not exists")
        
        if input_set is None or output_set is None:
            self.logger.error("Must provide both input and output sets")
            raise ValueError("Must provide both input and output sets")
        
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
        
    # -------------------------------
    # New: large-dataset ingestion + training
    # -------------------------------

    def prepare_large_dataset(
            self,
            input_file_list,
            output_file_list,
            h5_out_input: str,
            h5_out_output: str,
            max_signals_per_file: int | None = None,
            sample_signal: np.ndarray | None = None,
            shuffle=True
    ):
        """
        Convert many signal set files into two shuffled HDF5 datasets (input + output)
        with deterministic shuffling across both.

        If `sample_signal` is provided, its shape is used to determine an optimal
        HDF5 chunk size for faster I/O and model training throughput.

        Args:
            input_file_list:  list of paths to large input .npy sets
            output_file_list: list of paths to output .npy sets (same number of samples)
            h5_out_input:     destination HDF5 for inputs
            h5_out_output:    destination HDF5 for outputs
            max_signals_per_file: optional cap for subsampling each file
            sample_signal:    if given, used to determine chunk size for HDF5
            shuffle:          shuffle entire combined dataset
            random_state:     RNG seed or instance
        """

        # --- Determine chunk shape if sample signal is given ---
        # HDF5 chunking works best when a chunk contains a small batch of whole signals.
        # A reasonable default is (batch_of_128, signal_length...)
        if sample_signal is not None:
            sample_signal = np.asarray(sample_signal)
            # Choose chunk batch dimension based on signal size
            # Smaller signals → bigger batch chunks, larger signals → smaller batch chunks
            item_bytes = sample_signal.nbytes
            target_chunk_bytes = 1024 * 1024  # ~1MB per chunk (tunable)
            batch = max(1, target_chunk_bytes // item_bytes)
            chunk_shape_in = (batch, *sample_signal.shape)
            # Output is assumed real, same length
            chunk_shape_out = (batch,) if sample_signal.ndim == 1 else (batch, *sample_signal.shape[:-1])
        else:
            chunk_shape_in = None
            chunk_shape_out = None

        # --- First pass: count total samples ---
        total_samples = 0
        for in_file in input_file_list:
            n = np.load(in_file, mmap_mode='r').shape[0]
            if max_signals_per_file:
                n = min(n, max_signals_per_file)
            total_samples += n

        # --- Create HDF5 datasets ---
        with h5py.File(h5_out_input, 'w') as hf_in, h5py.File(h5_out_output, 'w') as hf_out:

            dset_in = hf_in.create_dataset(
                "X",
                shape=(total_samples, *sample_signal.shape) if sample_signal is not None else None,
                maxshape=(None,) + (() if sample_signal is None else sample_signal.shape),
                dtype=sample_signal.dtype if sample_signal is not None else 'complex64',
                chunks=chunk_shape_in
            )

            # Infer real output dataset shape
            if sample_signal is not None:
                out_shape = () if sample_signal.ndim == 1 else sample_signal.shape[:-1]
            else:
                out_shape = ()

            dset_out = hf_out.create_dataset(
                "y",
                shape=(total_samples, *out_shape),
                maxshape=(None,) + out_shape,
                dtype='float32',
                chunks=chunk_shape_out
            )

            # --- Streaming insert with optional shuffle-buffer ---
            order = np.arange(total_samples)
            if shuffle:
                self.rng.shuffle(order)

            write_pos = 0

            for in_file, out_file in zip(input_file_list, output_file_list):
                X_block = np.load(in_file, mmap_mode='r')
                y_block = np.load(out_file, mmap_mode='r')

                n_block = X_block.shape[0]
                if max_signals_per_file:
                    n_block = min(n_block, max_signals_per_file)

                sel = order[write_pos:write_pos + n_block]

                dset_in[sel] = X_block[:n_block]
                dset_out[sel] = y_block[:n_block]

                write_pos += n_block

            hf_in.flush()
            hf_out.flush()


    def train_on_hdf5(
            self,
            h5_input_path: str,
            h5_output_path: str,
            test_fraction: float = 0.1,
            batch_size: int = 32,
            epochs: int = 50,
            random_state=None
    ):
        """
        Train the existing Keras model on the prepared HDF5 datasets.
        No shuffling is required here if prepare_large_dataset already shuffled.
        """

        import h5py
        from tensorflow.keras.utils import Sequence

        class H5Sequence(Sequence):
            def __init__(self, h5_x, h5_y, indices, batch_size):
                self.h5_x = h5_x
                self.h5_y = h5_y
                self.indices = indices
                self.batch_size = batch_size

            def __len__(self):
                return len(self.indices) // self.batch_size

            def __getitem__(self, i):
                idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
                return self.h5_x[idx], self.h5_y[idx]

        hf_x = h5py.File(h5_input_path, 'r')
        hf_y = h5py.File(h5_output_path, 'r')

        X = hf_x["X"]
        y = hf_y["y"]
        N = X.shape[0]

        # Deterministic split index
        split = int(N * (1 - test_fraction))
        train_idx = np.arange(0, split)
        test_idx  = np.arange(split, N)

        train_seq = H5Sequence(X, y, train_idx, batch_size)
        test_seq  = H5Sequence(X, y, test_idx,  batch_size)

        self.model.fit(
            train_seq,
            validation_data=test_seq,
            epochs=epochs,
            workers=4,
            use_multiprocessing=True
        )

        hf_x.close()
        hf_y.close()
 
    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_log_params(self):        
        return self.log_params
    
    
    def get_training_params(self):
        return self.training_params
    
    
    def get_model_file_path(self):
        return self.model_file_path
    
    
    def get_config_name(self):
        return self.config_name
    
    
    def get_mlp_params(self):
        mlp_params ={
            "log_params": self.log_params,
            "training_params": self.training_params,
            "model_file_path": self.model_file_path,
            "config_name": self.config_name
        }
        return mlp_params