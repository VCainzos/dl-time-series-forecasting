import tensorflow as tf
from keras import backend


def mean_absolute_error_denor(std, mean):
    # Wrapper
    def mean_absolute_error_denor(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred * std + mean)
        y_true = tf.cast(y_true * std + mean, y_pred.dtype)

        return backend.mean(tf.abs(y_pred - y_true), axis=-1)

    class MeanAbsoluteErrorDenor(tf.keras.metrics.MeanMetricWrapper):
        def __init__(self, name="mean_absolute_error_denor", dtype=None):
            super().__init__(mean_absolute_error_denor, name, dtype=dtype)

    return MeanAbsoluteErrorDenor
