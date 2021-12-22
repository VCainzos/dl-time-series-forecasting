import tensorflow as tf
from .metrics.custom_metrics import mean_absolute_error_denor


def build_model_lstm(window):
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

    def model_lstm(hp):
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
        # Shape => [batch, 1,  lstm_units]
        lstm_model.add(
            tf.keras.layers.LSTM(
                hp.Int("units_lstm_" + str(layer + 1), 32, 40, step=2),
                return_sequences=False,
            )
        )
        # Shape => [batch, 1,  out_steps*features] (features=1 --> Capacity)
        lstm_model.add(tf.keras.layers.Dense(OUT_STEPS))
        # Shape => [batch, out_steps, features]
        lstm_model.add(tf.keras.layers.Reshape([OUT_STEPS, 1]))

        lstm_model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.metrics.MeanSquaredError(),
                mean_absolute_error_denor(
                    window.train_std, window.train_mean
                )(),
            ],
        )
        return lstm_model

    return model_lstm
