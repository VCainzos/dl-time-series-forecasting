import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing.preprocessing import *


class WindowGenerator:
    """Class to create sliding windows

    .. warning::
     param ``label_columns`` expects a list. More than one target feature is supported,
     however functions in :mod:`~visualization.visualization` may not work as long as they are defined to deal
     with 1-feature-dimension prediction tensors.
    """

    def __init__(
        self, df, input_width, label_width, shift, label_columns, **kwargs
    ):
        """Defines attributes for the sliding window object

        :param df: raw data
        :type df: pandas.DataFrame
        :param input_width: window size
        :type input_width: int
        :param label_width: prediction horizon
        :type label_width: int
        :param shift: offset, difference between last prediction and last input
        :type shift: int
        :param label_columns: tags of the features to predict
        :type label_columns: list
        """
        self.name = str(input_width)
        self.set_batch_size()  # Initialize the batch_size
        # Store the raw data.
        train, test = split(df, **kwargs)
        # And standarized dataframes
        self.train_df, self.test_df = standarize(train, test)
        # Full dataset to predict at last
        full = pd.concat([self.train_df, self.test_df])
        self.full_df = full
        # These both wil be used in custom metrics as variables
        self.train_std = train.std()
        self.train_mean = train.mean()
        # self.val_df = val_df

        # Work out the label column-indices as pairs key-value of a dictionary.
        self.label_columns = label_columns  # (used in split_window)
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }

        # Get the features-indices as pairs key-value of a dictionary (used in function split_window)
        self.column_indices = {
            name: i for i, name in enumerate(self.train_df.columns)
        }

        # Work out the window parameters.
        self.input_width = input_width  # Number of input values
        self.label_width = label_width  # Number of predictions
        self.shift = shift  # This is the difference between the last input and label indices (time into the future -> offset)

        # (not all the indices through the window must be used in every case)
        self.total_window_size = input_width + shift

        # Returns a slice object representing the set of indices specified by range(start, stop, step)
        self.input_slice = slice(0, input_width)
        # Create an array of slicing input indices
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        # Get the label indices
        self.label_start = self.total_window_size - self.label_width
        # Create slice object
        self.labels_slice = slice(self.label_start, None)
        # Create an array of slicing label indices
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice
        ]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        """Splits a raw tensor of samples in inputs and labels using arrays of indices

        :param features: tensor of time-sequences
        :type features: tf.data
        :return: dataset tensor of inputs and labels
        :rtype: tuple

        .. note::
            This function is used to map tensor into inputs and labels in
            :attr:`make_dataset`.

        """

        # Split tensor using inputs indices across samples dimension
        inputs = features[:, self.input_slice, :]
        # Split tensor using label indices across samples dimension
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            # Split tensor using label column name across features dimension
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes manually.
        # This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def set_batch_size(self, batch_size=32):
        """Sets batch_size attribute value

        :param batch_size: number of samples per batch, defaults to 32
        :type batch_size: int, optional
        """
        self.batch_size = batch_size

    def make_dataset(self, data):
        """Takes a dataframe and build a tensor dataset of time-sequences

        :param data: dataframe with time series
        :type data: pandas.DataFrame
        :return: dataset tensor of inputs and labels
        :rtype: tf.data
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,  # Using label width as stride to avoid overlapping predictions
            shuffle=False,  # Doing shuffle here would cause the loss of data sequence missing the tendency, this is not desired
            batch_size=self.batch_size,  # Number of time sequences in each batch
        )
        ds = ds.map(self.split_window)

        return ds  # Dataset = batch/'s of tensors

    @property
    def train(self):
        """Property to deal with training dataset

        :return: training dataset of inputs and labels
        :rtype: tf.data
        """
        ds = self.make_dataset(self.train_df)
        # Shuffle all samples

        samples = len([i for i in ds.unbatch().batch(1)])
        """
        ds.unbatch().batch(samples).shuffle(
            samples, reshuffle_each_iteration=False
        )
        ds.unbatch().batch(self.batch_size)
        """

        # Shuffle data before building the Dataset (shuffle method shuffles among batches)
        return (
            ds.unbatch()
            .batch(1)
            .shuffle(len(samples), reshuffle_each_iteration=False)
            .unbatch()
            .batch(self.batch_size)
        )  # .shuffle(len(ds), reshuffle_each_iteration=False)

    # @property
    # def val(self):
    # return self.make_dataset(self.val_df)

    @property
    def test(self):
        """Property to deal with test dataset

        :return: test dataset of inputs and labels
        :rtype: tf.data
        """
        return self.make_dataset(self.test_df)

    @property
    def full(self):
        """Property to deal with test dataset

        :return: test dataset of inputs and labels
        :rtype: tf.data
        """
        return self.make_dataset(self.full_df)

    def save_model(self, model):
        """Save fitted models in the window object to access easily to them

        :param model: optimized and trained model
        :type model: tf.keras.Model
        """
        self.models = getattr(self, "models", {})
        self.models[model.name] = model

    def save_performance(self, model, training, evaluation):
        """Tracks model performance

        :param model: optimized and trained model
        :type model: tf.keras.Model
        :param training: training evaluation results
        :type training: dict
        :param evaluation: test evaluation results
        :type evaluation: dict
        """
        self.multi_train_performance = getattr(
            self, "multi_train_performance", {}
        )
        self.multi_performance = getattr(self, "multi_performance", {})

        self.multi_train_performance[
            model.name + "<sub>" + self.name + "</sub>"
        ] = dict(zip(model.metrics_names, training))
        self.multi_performance[
            model.name + "<sub>" + self.name + "</sub>"
        ] = dict(zip(model.metrics_names, evaluation))
