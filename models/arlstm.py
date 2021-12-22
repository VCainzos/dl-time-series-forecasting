import tensorflow as tf
from .metrics.custom_metrics import mean_absolute_error_denor


class FeedBack(tf.keras.Model):
    """Class to customized Auto-regressive lstm model behaviour"""

    def __init__(self, hp, window):
        """Define here the architecture of the network

        :param hp: argument to define the hyperparameters, it is passed automatically by the oracle
        :type hp: kt.engine.hyperparameters.HyperParameters
        :param window: window object
        :type window: WindowGenerator
        """

        super().__init__(name="ARLSTM")
        self.outsteps = window.label_width  # Get output width
        warmup_layers = hp.Int("warmup_layers", 0, 3, default=0)
        units = hp.Int("units", 32, 40, step=2)
        # out_steps = 2
        # self.out_steps = out_steps
        self.units = units
        # It is necessary for units to be fitted with the previous layer when operating at cell level (*)
        # Also, use another cell because it will be computing different input_dim (time_steps)
        self.lstm_cell = tf.keras.layers.LSTMCell(units)

        # Stacking layers during warming-up to achieve a greater level of abstraction
        self.lstm_rnn = []
        # This is the previous layer of the cell level computation (*)
        self.lstm_rnn_warmup = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(units), return_state=True
        )
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        for layer in range(warmup_layers):
            self.lstm_rnn.append(
                tf.keras.layers.RNN(
                    tf.keras.layers.LSTMCell(
                        hp.Int("units_rnn_" + str(layer), 32, 40, step=2)
                    ),
                    return_sequences=True,
                )
            )
        # Once the warmup is completed, it gives only one output throughout the whole context of the input sample (sequences of data)
        self.dense = tf.keras.layers.Dense(1)

    def warmup(self, inputs):
        """Pre-computation of the inputs after getting horizon predictions

        :param inputs: tensor of inputs
        :type inputs: tf.Tensor
        :return: prediction and state from the last time-step
        :rtype: tuple
        """

        # inputs.shape => (batch, time, features)

        x_in = inputs
        y_out = inputs

        for layer in self.lstm_rnn:
            y_out = layer(x_in)
            x_in = y_out

        # x.shape => (batch, lstm_units) return_sequences=False
        x, *state = self.lstm_rnn_warmup(y_out)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        """Dealing with inputs to get output predictions

        :param inputs: tensor of inputs
        :type inputs: tf.Tensor
        :param training: whether or not to set trainable parameters, defaults to None
        :type training: bool, optional
        :return: output tensor of predictions
        :rtype: tf.Tensor
        """

        # out_steps=inputs[1].shape[0] #Get the size of each sample/serie.
        # Output shape equal to input for this configuration (automatically, when the model is called)
        out_steps = self.outsteps

        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)
        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


def build_model_Feedback(window):
    """Wrapper function which passes the window object to the creation function of the Keras model

    :param window: window object
    :type window: WindowGenerator
    :return: creation function of the Keras model
    :rtype: function

    .. note::
       The function :func:`~model_Feedback` that creates and returns a cutomized auto-regressive
       lstm model is nested within the Wrapper.

    .. function:: model_Feedback(hp)

        Function that creates and returns an auto-regressive lstm model

        :param hp: argument to define the hyperparameters, it is passed automatically by the oracle
        :type hp: kt.engine.hyperparameters.HyperParameters
        :return: Keras model
        :rtype: FeedBack
    """

    def model_Feedback(hp):
        """Function that creates and returns an auto-regressive lstm model

        :param hp: argument to define the hyperparameters, it is passed automatically by the oracle
        :type hp: kt.engine.hyperparameters.HyperParameters
        :return: Keras model
        :rtype: FeedBack
        """

        Feedback_model = FeedBack(hp, window)
        Feedback_model.compile(
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
        return Feedback_model

    return model_Feedback
