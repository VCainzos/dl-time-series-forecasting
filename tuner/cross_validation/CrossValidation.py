from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np


class CrossValidation:
    """class to implement cross-validation used in KerasTuner"""

    def __init__(self, epochs=5, batch_size=32, folds=3, shuffle=True):
        """Defines main attributes for the cross-validation

        Args:
            epochs (int, optional): times data will be processed. Defaults to 5.
            batch_size (int, optional): data processed after which weights will be updated. Defaults to 32.
            folds (int, optional): partitions of the training set used for cross-validation. Defaults to 3.
            shuffle (bool, optional): specifies whether or not shuffle data. Defaults to True.
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

        Args:
            historial (dict): dictionary with metrics and results

        Returns:
            means (dict): dictionary with metrics and mean results
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

        Args:
            model (tf.keras.Model): model to perform cross-validation
            dataset (tf.data): dataset tensor with inputs and labels

        Returns:
            results (dict): dictionary with metrics and mean results
        """

        self.build()  # Call build method

        samples = len([i for i in dataset.unbatch().batch(1)])

        x = next(iter(dataset.unbatch().batch(samples)))[0].numpy()
        y = next(iter(dataset.unbatch().batch(samples)))[1].numpy()

        historial = []

        model_aux = model

        for fold, (train_indices, val_indices) in enumerate(
            self.kf.split(next(iter(dataset.unbatch().batch(samples)))[0])
        ):

            model = model_aux
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
        results = self.get_means(historial)
        return results
