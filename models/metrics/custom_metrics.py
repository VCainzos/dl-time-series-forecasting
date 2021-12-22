import tensorflow as tf
from keras import backend


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
