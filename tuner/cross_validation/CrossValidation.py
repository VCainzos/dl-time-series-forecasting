from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np


class CrossValidation:
    """class to implement cross-validation used in KerasTuner"""

    def __init__(self, epochs=50, batch_size=32, folds=5, shuffle=True):
        """Defines main attributes for the cross-validation

        :param epochs: times data will be processed, defaults to 5
        :type epochs: int, optional
        :param batch_size: data processed after which weights will be updated, defaults to 32
        :type batch_size: int, optional
        :param folds: partitions of the training set used for cross-validation, defaults to 3
        :type folds: int, optional
        :param shuffle: specifies whether or not shuffle data, defaults to True
        :type shuffle: bool, optional
        """

        self.epochs = epochs
        self.batch_size = batch_size
        self.folds = folds
        self.shuffle = shuffle

    def build(self):
        """Creates the CrossValidation object"""
        self.kf = KFold(self.folds, shuffle=self.shuffle)

    def get_means(self, historial):
        """Manages the means of the historial results used to plot learning curves

        :param historial: dictionary with metrics and results
        :type historial: dict
        :return: dictionary with metrics and mean resulst
        :rtype: dict
        """

        means = {}
        # Loop for metrics
        for key in historial[0].history.keys():
            key_values = []
            # Loop for histories
            for i in historial:
                key_values.append(i.history[key])
            means[key] = np.mean(key_values, axis=0)
        return means

    def __call__(self, model, dataset, *args, **kwargs):
        """Defines the inner computation of the cross-validation

        :param model: model to perform cross-validation
        :type model: tf.keras.Model
        :param dataset: dataset tensor with inputs and labels
        :type dataset: tf.data
        :return: dictionary with metrics and mean results
        :rtype: dict
        """

        self.build()  # Call build method

        samples = len([i for i in dataset.unbatch().batch(1)])

        x = next(iter(dataset.unbatch().batch(samples)))[0].numpy()
        y = next(iter(dataset.unbatch().batch(samples)))[1].numpy()

        historial = []
        kscores = []

        for fold, (train_indices, val_indices) in enumerate(
            self.kf.split(next(iter(dataset.unbatch().batch(samples)))[0])
        ):
            # Reset model weights at the begining of each fold
            model.reset_states()

            print(f"Fold {fold}")
            x_train = tf.stack(x[train_indices, :, :])
            y_train = tf.stack(y[train_indices, :, :])
            dataset_train = (
                tf.data.Dataset.from_tensors((x_train, y_train))
                .unbatch()
                .batch(self.batch_size)
            )

            x_val = tf.stack(x[val_indices, :, :])
            y_val = tf.stack(y[val_indices, :, :])
            dataset_val = (
                tf.data.Dataset.from_tensors((x_val, y_val))
                .unbatch()
                .batch(self.batch_size)
            )

            history = model.fit(
                dataset_train,
                epochs=self.epochs,
                validation_data=dataset_val,
                *args,
                **kwargs,
            )
            # historial.append(history.history['val_mean_absolute_error'])
            historial.append(history)

            # Evaluation on validation set
            kscore = model.evaluate(dataset_val, verbose=0)
            kscores.append(kscore)

        results = self.get_means(historial)

        return results, kscores
