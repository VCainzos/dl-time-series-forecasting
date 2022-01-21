def split(df, train=0.8):
    """Divides raw data into training and test set

    :param df: raw data
    :type df: pandas.DataFrame
    :param train: training set threshold, defaults to 0.9
    :type train: float, optional
    :return: training and test sets
    :rtype: tuple

    .. warning::
       Validation set is not required nor supported as long as cross-validation is used.
    """

    n = len(df)
    train_df = df[0 : int(n * train)]
    # val_df = df[int(n*train):int(n*val)]
    test_df = df[int(n * train) :]

    return train_df, test_df


def standarize(train_df, test_df):
    """Centers data with mean 0 and standard deviation 1

    .. math::
            \\mathsf{x_{nor}} =
            {\\mathsf{x} - \mu \\over  \sigma}

    :param train_df: training data
    :type train_df: pandas.DataFrame
    :param test_df: test data
    :type test_df: pandas.DataFrame
    :return: standarized training and test set
    :rtype:

    .. note::
        Mean and standard deviation of the training set is used to center splits,
        in order to put test data information aside until the evalutation stage is done.
    """

    # Calculate mean and standard deviation of the training set
    train_mean = train_df.mean()
    train_std = train_df.std()

    # Standarization of the data throughout the mean and standard deviation across train_df
    train_df = (train_df - train_mean) / train_std
    # val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, test_df
