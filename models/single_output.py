import tensorflow as tf
import tensorflow_addons as tfa
from .metrics.custom_metrics import (
    mean_absolute_error_denor,
    mean_absolute_percentage_error,
    r_square_error,
)


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__(name=model.name)
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return delta  # + inputs[:, :, :1]  # Inputs shape: (batch_size, time, Pmed)


def build_model_single_output(window):
    """Wrapper function which passes the window object to the creation function of the Keras model

    :param window: window object
    :type window: WindowGenerator
    :return: creation function of the Keras model
    :rtype: function

    .. note::
       The function :func:`~model_lstm` that creates and returns a customized lstm model is nested
       within the Wrapper.

    .. function:: model_lstm(hp)

        Function that creates and returns an lstm model

        :param hp: argument to define the hyperparameters, it is passed automatically by the oracle during model creation
        :type hp: kt.engine.hyperparameters.HyperParameters
        :return: Keras model
        :rtype: tf.keras.Model
    """

    def model_single_output(hp):
        """Function that creates and returns an lstm model

        :param hp: argument to define the hyperparameters, it is passed automatically by the oracle during model creation
        :type hp: kt.engine.hyperparameters.HyperParameters
        :return: Keras model
        :rtype: tf.keras.Model
        """

        lstm_layers = hp.Int("lstm_layers", 1, 3)
        OUT_STEPS = window.label_width
        lstm_model = tf.keras.models.Sequential(name="LSTM")
        # Shape [batch, time, features] => [batch, time, lstm_units]
        for layer in range(lstm_layers):
            # Stacking lstm layers
            lstm_model.add(
                tf.keras.layers.LSTM(
                    hp.Int("units_lstm_" + str(layer), 32, 40, step=2),
                    return_sequences=True,
                )
            )
        # Shape => [batch, time,  lstm_units]
        lstm_model.add(
            tf.keras.layers.LSTM(
                hp.Int("units_lstm_" + str(layer + 1), 32, 40, step=2),
                return_sequences=True,
            )
        )
        # Shape => [batch, time,  features] (features=1 --> Capacity)
        lstm_model.add(
            tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.zeros())
        )

        # residual_lstm_model = ResidualWrapper(lstm_model)

        lstm_model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.metrics.MeanSquaredError(),
                tf.metrics.MeanAbsolutePercentageError(),
                # mean_absolute_percentage_error(
                #    window.train_std, window.train_mean
                # )(),
                # tfa.metrics.r_square.RSquare(y_shape=(OUT_STEPS, 1)),
                r_square_error(window.train_std, window.train_mean)(),
                mean_absolute_error_denor(
                    window.train_std, window.train_mean
                )(),
            ],
            run_eagerly=True,
        )
        return lstm_model

    return model_single_output
