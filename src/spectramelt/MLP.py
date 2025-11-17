import numpy as np
from .utils import load_config_from_json, get_logger
import tensorflow as tf
import h5py
import time
import keras
from keras import (
    layers,
    losses,
    backend,
    Sequential,
    optimizers,
    Input
)
from keras.callbacks import EarlyStopping
from keras.activations import get as get_activation
from keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split

@keras.saving.register_keras_serializable(package="CustomLosses")
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

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
                 model_params=None,
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
            mlp_params['training_params'] = training_params
            mlp_params['config_name'] = config_name
            mlp_params['log_params'] = log_params

        self.set_mlp_params(mlp_params)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")

        # Ensure custom losses are registered
        get_custom_objects().update({"root_mean_squared_error": root_mean_squared_error})
 
    # -------------------------------
    # Setters
    # -------------------------------
        
    def set_mlp_params(self, mlp_params=None):
        if mlp_params is None:
            mlp_params = {}
        model_params = mlp_params.get('model_params', None)
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
        
        self.set_model_params(model_params)
        self.set_training_params(training_params)
        self.set_config_name(config_name)
        
    
    def set_config_name(self, config_name):
        if config_name is None:
            config_name = "MLP_Config_1"
        self.config_name = config_name
        

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
                "shuffle": True
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
  
        early_stopping_params = training_params.get('early_stopping', {})
        monitor = early_stopping_params.get('monitor', "val_loss")
        if not self.is_valid_monitor(monitor):
            self.logger.error(f"{monitor} is not a valid Keras monitor value")
            raise ValueError(f"{monitor} is not a valid Keras monitor value")           
        
        self.training_params = training_params

    # -------------------------------
    # Core functional methods
    # -------------------------------

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
        backend.clear_session()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.keras.backend.clear_session()


    def create_model(self, input_signal_size, output_signal_size, model_file_path=None):
        if model_file_path is None:
            model_file_path = self.model_params.get('file_path', "ml_model.keras")
            
        loss_type = self.training_params.get('loss_type', "mean_squared_error")
        learning_rate = self.training_params.get('learning_rate', 0.00001)
        hidden_layer_width_mult = self.model_params.get('hidden_layer_width_mult', [1])
        activation_functions = self.model_params.get('activation_per_layer',
                                                     ["linear", "linear"])

        mlp_model = Sequential()
        mlp_model.add(Input(shape=(input_signal_size,),
                            name="mlp_model_in"))
        
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
        mlp_model.compile(optimizer=mlp_opt, loss=get_custom_objects().get(loss_type, loss_type))
        mlp_model.save(model_file_path, overwrite=True)
        

    def load_model(self, model_file_path=None):
        if model_file_path is None:
            model_file_path = self.model_params.get('file_path', "ml_model.keras")

        if not model_file_path.exists():
            self.logger.error(f"Model file {model_file_path} does not exist")
            raise ValueError(f"Model file {model_file_path} does not exist")


        # Load model normally; custom losses are already registered
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
            sample_signal: np.ndarray | None = None
    ):
        """
        Convert multiple .npy signal files into two large HDF5 datasets (X and y),
        written sequentially for speed and then globally shuffled in-place.
        """

        # ------------------------------------------------------------
        # 1. Determine chunk sizes if sample_signal is provided
        # ------------------------------------------------------------
        if sample_signal is not None:
            sample_signal = np.asarray(sample_signal)

            item_bytes = sample_signal.nbytes
            target_chunk_bytes = 1 * 1024 * 1024  # 1 MB
            batch = max(1, target_chunk_bytes // item_bytes)

            chunk_shape_in = (batch, *sample_signal.shape)

            # Output is real-valued, same length as input signal (your assumption)
            if sample_signal.ndim == 1:
                out_shape = (sample_signal.shape[0],)
                chunk_shape_out = (batch, sample_signal.shape[0])
            else:
                out_shape = sample_signal.shape[:-1]
                chunk_shape_out = (batch, *out_shape)

            input_dtype = sample_signal.dtype
        else:
            # Fallback if no sample provided
            self.logger.error("sample_signal must be provided for shape inference.")
            raise ValueError("sample_signal must be provided for shape inference.")

        # ------------------------------------------------------------
        # 2. First pass: count number of total samples
        # ------------------------------------------------------------
        total_samples = 0
        for in_file in input_file_list:
            n = np.load(in_file, mmap_mode='r').shape[0]
            if max_signals_per_file:
                n = min(n, max_signals_per_file)
            total_samples += n

        self.logger.info(f"Total samples across all files: {total_samples}")

        # ------------------------------------------------------------
        # 3. Create HDF5 files and datasets
        # ------------------------------------------------------------
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

            # --------------------------------------------------------
            # 4. Sequential write — NO FANCY INDEXING
            # --------------------------------------------------------
            write_pos = 0
            for in_file, out_file in zip(input_file_list, output_file_list):

                X_block = np.load(in_file, mmap_mode="r")
                y_block = np.load(out_file, mmap_mode="r")

                n_block = X_block.shape[0]
                if max_signals_per_file:
                    n_block = min(n_block, max_signals_per_file)

                self.logger.debug(f"[WRITE] {in_file}  → indices [{write_pos}:{write_pos+n_block}]")

                # Write sequentially — ALWAYS SAFE, ALWAYS FAST
                dset_in[write_pos:write_pos + n_block] = X_block[:n_block]
                dset_out[write_pos:write_pos + n_block] = y_block[:n_block]

                write_pos += n_block

            hf_in.flush()
            hf_out.flush()

        # ------------------------------------------------------------
        # 5. Global shuffle (deterministic)
        # ------------------------------------------------------------
        self.logger.info("Applying deterministic global shuffle…")

        # Generate permutation
        perm = self.rng.permutation(total_samples)

        with h5py.File(h5_out_input, "r+") as hf_in, \
            h5py.File(h5_out_output, "r+") as hf_out:

            X = hf_in["X"]
            y = hf_out["y"]

            # Create temporary datasets
            Xtmp = hf_in.create_dataset(
                "X_shuffled",
                shape=X.shape,
                dtype=X.dtype,
                chunks=X.chunks
            )
            ytmp = hf_out.create_dataset(
                "y_shuffled",
                shape=y.shape,
                dtype=y.dtype,
                chunks=y.chunks
            )

            # Shuffle in manageable batches
            batch = 8192
            for start in range(0, total_samples, batch):
                end = min(start + batch, total_samples)
                
                idx = perm[start:end]        # unsorted permutation
                sorted_idx = np.sort(idx)    # h5py requires increasing order

                # Read block using sorted indices
                X_block_sorted = X[sorted_idx]
                y_block_sorted = y[sorted_idx]

                # Map sorted block back into true perm order
                # inverse map of argsort
                inv_order = np.argsort(np.argsort(idx))

                # Reorder to match the permutation
                X_block = X_block_sorted[inv_order]
                y_block = y_block_sorted[inv_order]

                # Write back into [start:end]
                Xtmp[start:end] = X_block
                ytmp[start:end] = y_block

            # Replace original datasets
            del hf_in["X"]
            del hf_out["y"]
            hf_in.move("X_shuffled", "X")
            hf_out.move("y_shuffled", "y")

        self.logger.info("[DONE] Dataset creation and shuffle complete.")


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
    

    def train_on_hdf5(self, h5_input_path, h5_output_path, model_file_path=None):
        """
        Train the existing Keras model on large HDF5 datasets using efficient batched loading.
        Splits train/test deterministically before batching to avoid empty datasets.
        """

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

        # Build HDF5 loader
        get_batch, total_num_sigs = self.make_hdf5_batch_loader(h5_input_path, h5_output_path)

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
    
    
    def get_training_params(self):
        return self.training_params
    
    
    def get_model_params(self):
        return self.model_params
    
    
    def get_config_name(self):
        return self.config_name
    
    
    def get_mlp_params(self):
        mlp_params ={
            "log_params": self.log_params,
            "training_params": self.training_params,
            "model_params": self.model_params,
            "config_name": self.config_name
        }
        return mlp_params