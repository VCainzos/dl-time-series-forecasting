import tensorflow as tf
from .metrics.custom_metrics import mean_absolute_error_denor

def build_model_lstm(window):
    def model_lstm(hp):
        lstm_layers = hp.Int("lstm_layers", 1, 3)
        OUT_STEPS=window.label_width
        lstm_model = tf.keras.models.Sequential(name='LSTM')
        # Shape [batch, time, features] => [batch, time, lstm_units]
        for layer in range(lstm_layers):
            lstm_model.add(tf.keras.layers.LSTM(hp.Int('units_lstm_'+str(layer), 32, 40, step=2), return_sequences=True)) #Stacking lstm layers
        # Shape => [batch, 1,  lstm_units]
        lstm_model.add(tf.keras.layers.LSTM(hp.Int('units_lstm_'+str(layer+1), 32, 40, step=2), return_sequences=False))
        # Shape => [batch, 1,  out_steps*features] (features=1 --> Capacity)
        lstm_model.add(tf.keras.layers.Dense(OUT_STEPS))
        # Shape => [batch, out_steps, features]
        lstm_model.add(tf.keras.layers.Reshape([OUT_STEPS, 1]))
        
        lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError(), mean_absolute_error_denor(window.train_std, window.train_mean)()])
        return lstm_model
    return model_lstm