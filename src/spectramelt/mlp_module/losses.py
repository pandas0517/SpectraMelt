import tensorflow as tf 
from tensorflow.keras.utils import get_custom_objects
import keras
from keras import (
    losses
)


@keras.saving.register_keras_serializable(package="CustomLosses")
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


@keras.saving.register_keras_serializable(package="CustomLosses")
class HuberSparseAmplitudeLoss(keras.losses.Loss):
    def __init__(
        self,
        tau=0.01,
        amplitude_weight=0.1,
        delta=1.0,
        name="huber_sparse_amplitude_loss",
    ):
        super().__init__(name=name)
        self.tau = tau
        self.amplitude_weight = amplitude_weight
        self.delta = delta
        self.huber = losses.Huber(
            delta=self.delta,
            reduction=losses.Reduction.NONE
        )

    def noise_aware_weight(self, y_true):
        eps = tf.constant(1e-8, dtype=y_true.dtype)
        return tf.sigmoid((tf.abs(y_true) - self.tau) / (0.1 * self.tau + eps))

    def call(self, y_true, y_pred):
        # ---- Noise-aware Huber shape loss ----
        w = self.noise_aware_weight(y_true)
        huber_vals = self.huber(y_true, y_pred)
        weighted_shape = tf.reduce_mean(w * huber_vals)

        # ---- Sparse amplitude penalty (L1) ----
        true_amp = tf.reduce_sum(tf.abs(y_true), axis=-1)
        pred_amp = tf.reduce_sum(tf.abs(y_pred), axis=-1)

        amp_ratio = pred_amp / (true_amp + 1e-8)
        amp_penalty = tf.reduce_mean(tf.abs(amp_ratio - 1.0))

        return weighted_shape + self.amplitude_weight * amp_penalty

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tau": self.tau,
                "amplitude_weight": self.amplitude_weight,
                "delta": self.delta,
            }
        )
        return config


def resolve_loss(loss_type, loss_params=None):
    """
    Resolves a loss from a string or callable.
    If loss_type is a registered custom class, instantiate it with loss_params.
    """
    if loss_params is None:
        loss_params = {}

    # Check if it's a string referring to a registered custom loss
    if isinstance(loss_type, str):
        # First, see if it's a custom object
        custom_cls = get_custom_objects().get(loss_type)
        if custom_cls is not None:
            return custom_cls(**loss_params)  # instantiate with params

        # Otherwise, fallback to Keras built-in losses
        try:
            return losses.get(loss_type)
        except ValueError:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    # If already a callable/loss instance, just return
    return loss_type