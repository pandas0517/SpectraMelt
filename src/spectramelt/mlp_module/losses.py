import tensorflow as tf 
from keras.utils import get_custom_objects
import keras
from keras import (
    losses
)
import inspect


@keras.saving.register_keras_serializable(package="CustomLosses")
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


@keras.saving.register_keras_serializable(package="CustomLosses")
def weighted_root_mean_squared_error(
    y_true,
    y_pred,
    alpha=5.0,
    eps=1e-6
):
    """
    Weighted RMSE that emphasizes non-zero targets.
    
    alpha : strength of weighting for non-zero targets
    eps   : threshold to detect sparsity
    """

    # Identify non-zero (pulse) locations
    nonzero_mask = tf.cast(tf.abs(y_true) > eps, tf.float32)

    # Weight: 1 for zeros, (1 + alpha) for non-zeros
    weights = 1.0 + alpha * nonzero_mask

    squared_error = tf.square(y_pred - y_true)

    weighted_mse = tf.reduce_mean(weights * squared_error)

    return tf.sqrt(weighted_mse)


@keras.saving.register_keras_serializable(package="CustomLosses")
def weighted_rmse_with_energy(
    y_true,
    y_pred,
    alpha=5.0,
    gamma=0.1,
    eps=1e-6
):
    # --- Weighted RMSE (what already works) ---
    nonzero = tf.cast(tf.abs(y_true) > eps, tf.float32)
    weights = 1.0 + alpha * nonzero
    rmse = tf.sqrt(tf.reduce_mean(weights * tf.square(y_pred - y_true)))

    # --- Per-sample magnitude conservation ---
    true_energy = tf.reduce_sum(tf.abs(y_true), axis=-1)
    pred_energy = tf.reduce_sum(tf.abs(y_pred), axis=-1)

    energy_loss = tf.reduce_mean(tf.square(pred_energy - true_energy))

    return rmse + gamma * energy_loss


@keras.saving.register_keras_serializable(package="CustomLosses")
class HuberSparseAmplitudeLoss(keras.losses.Loss):
    def __init__(
        self,
        tau=0.01,
        amplitude_weight=0.1,
        delta=1.0,
        reduction=keras.losses.Reduction.AUTO,
        name="huber_sparse_amplitude_loss",
    ):
        super().__init__(reduction=reduction, name=name)

        self.tau = tau
        self.amplitude_weight = amplitude_weight
        self.delta = delta

        self.huber = losses.Huber(
            delta=self.delta,
            reduction=losses.Reduction.NONE
        )

    def noise_aware_weight(self, y_true):
        return tf.sigmoid((tf.abs(y_true) - self.tau) / (0.1 * self.tau))

    def call(self, y_true, y_pred):
        # ---- Noise-aware shape loss ----
        w = self.noise_aware_weight(y_true)
        shape_error = tf.abs(y_pred - y_true)
        weighted_shape = tf.reduce_mean(w * shape_error)

        # ---- Sparse amplitude penalty ----
        true_amp = tf.reduce_sum(tf.abs(y_true), axis=-1)
        pred_amp = tf.reduce_sum(tf.abs(y_pred), axis=-1)

        amp_ratio = pred_amp / (true_amp + 1e-8)
        amp_penalty = tf.reduce_mean(tf.abs(amp_ratio - 1.0))

        return weighted_shape + self.amplitude_weight * amp_penalty

    def get_config(self):
        config = super().get_config()
        config.update({
            "tau": self.tau,
            "amplitude_weight": self.amplitude_weight,
            "delta": self.delta,
        })
        return config
    

def _normalize_loss_name(name: str) -> str:
    """
    Normalize loss names for matching:
    - lowercase
    - remove underscores and hyphens
    """
    return name.lower().replace("_", "").replace("-", "")


def resolve_loss(loss_type, loss_params=None):
    """
    Resolve a loss from:
      - string (custom or keras builtin)
      - keras.losses.Loss instance
      - callable

    Returns a loss instance or callable suitable for model.compile().
    """
    if loss_params is None:
        loss_params = {}

    # --------------------------------------------------
    # Case 1: already a Loss instance or callable
    # --------------------------------------------------
    if not isinstance(loss_type, str):
        return loss_type

    requested = _normalize_loss_name(loss_type)

    # --------------------------------------------------
    # Case 2: custom registered losses (class or function)
    # --------------------------------------------------
    custom_objects = get_custom_objects()

    for name, obj in custom_objects.items():
        normalized_name = _normalize_loss_name(name.split(">")[-1])

        if normalized_name == requested:
            # Custom loss class → instantiate
            if inspect.isclass(obj) and issubclass(obj, losses.Loss):
                return obj(**loss_params)

            # Custom loss function
            return obj

    # --------------------------------------------------
    # Case 3: keras built-in losses
    # --------------------------------------------------
    try:
        return losses.get(loss_type)
    except ValueError:
        pass

    # --------------------------------------------------
    # Failure: give helpful diagnostics
    # --------------------------------------------------
    available_custom = sorted(
        name for name in custom_objects
        if ">" in name
    )

    raise ValueError(
        f"Unknown loss_type: '{loss_type}'.\n"
        f"Available custom losses: {available_custom}\n"
        f"Available keras losses: https://keras.io/api/losses/"
    )