import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend
import numpy as np


def mean_absolute_error_denor(std, mean):
    """Wrapper function that passes outside arguments to the customized metric

    :param std: standard deviation of the data
    :type std: float
    :param mean: mean of the data
    :type mean: float
    :return: customized metric class
    :rtype: MeanAbsoluteErrorDenor

    .. note::
       The customized metric function :func:`~mean_absolute_error_denor.mean_absolute_error_denor` and the subclass
       :class:`~mean_absolute_error_denor.MeanAbsoluteErrorDenor` to implemented it are nested within the Wrapper.

    .. function:: mean_absolute_error_denor.mean_absolute_error_denor(y_true, y_pred)

        Customized metric function

        :param y_true: real labels
        :type y_true: tf.Tensor
        :param y_pred: predicted labels
        :type y_pred: tf.Tensor
        :return: metric results
        :rtype: tf.Tensor

    .. class:: mean_absolute_error_denor.MeanAbsoluteErrorDenor(tf.keras.metrics.MeanMetricWrapper)

        Subclass to implement the customized metric
    """

    def mean_absolute_error_denor(y_true, y_pred):
        """Customized metric function

        :param y_true: real labels
        :type y_true: tf.Tensor
        :param y_pred: predicted labels
        :type y_pred: tf.Tensor
        :return: metric results
        :rtype: tf.Tensor
        """
        y_pred = tf.convert_to_tensor(y_pred * std + mean)
        y_true = tf.cast(y_true * std + mean, y_pred.dtype)

        return backend.mean(tf.abs(y_pred - y_true), axis=-1)

    class MeanAbsoluteErrorDenor(tf.keras.metrics.MeanMetricWrapper):
        """Subclass to implement the customized metric"""

        def __init__(self, name="mean_absolute_error_denor", dtype=None):
            super().__init__(mean_absolute_error_denor, name, dtype=dtype)

    return MeanAbsoluteErrorDenor


def mean_absolute_percentage_error(std, mean):
    def mean_absolute_percentage_error(y_true, y_pred):
        """Customized metric function

        :param y_true: real labels
        :type y_true: tf.Tensor
        :param y_pred: predicted labels
        :type y_pred: tf.Tensor
        :return: metric results
        :rtype: tf.Tensor
        """
        y_pred = tf.convert_to_tensor(y_pred * std + mean)
        y_true = tf.cast(y_true * std + mean, y_pred.dtype)

        return backend.mean(
            tf.math.divide_no_nan(
                tf.abs((y_true - y_pred) * 100), tf.abs(y_true)
            ),
            axis=-1,
        )

    class MeanAbsolutePercentageError(tf.keras.metrics.MeanMetricWrapper):
        """Subclass to implement the customized metric"""

        def __init__(self, name="mean_absolute_percentage_error", dtype=None):
            super().__init__(mean_absolute_percentage_error, name, dtype=dtype)

    return MeanAbsolutePercentageError


def r_square_error(std, mean):
    metric = tfa.metrics.r_square.RSquare()

    def r_square_error(y_true, y_pred):
        """Customized metric function

        :param y_true: real labels
        :type y_true: tf.Tensor
        :param y_pred: predicted labels
        :type y_pred: tf.Tensor
        :return: metric results
        :rtype: tf.Tensor
        """
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        total = tf.reduce_sum(
            tf.square(tf.subtract(y_true, tf.reduce_mean(y_true)))
        )

        r2 = tf.subtract(
            1.0, tf.math.divide_no_nan(residual, total)
        )  # divide_no_nan avoids zero-denominators
        return r2
        """
        metric.update_state(y_true.numpy().flatten(), y_pred.numpy().flatten())
        result = metric.result().numpy()
        metric.reset_state()
        return np.mean(result)

    class RSquareError(tf.keras.metrics.MeanMetricWrapper):
        """Subclass to implement the customized metric"""

        def __init__(self, name="r_square_error", dtype=None):
            super().__init__(r_square_error, name, dtype=dtype)

    return RSquareError
