def split(df, train=0.9, test=0.1):
    """Divides raw data into training and test set (validation set is not required as long as cross-validation)

    Args:
        df (pandas.DataFrame): dataframe containing raw data
        train (float, optional): rate of the training set. Defaults to 0.9.
        test (float, optional): rate of the test set. Defaults to 0.1.

    Returns:
        train_df, test_df (tuple): training and test set
    """
    n = len(df)
    train_df = df[0 : int(n * 0.9)]
    # val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n * 0.9) :]

    num_features = df.shape[1]
    return train_df, test_df


def standarize(train_df, test_df):
    """Centers data with mean 0 and standard deviation 1

    Args:
        train_df (pandas.DataFrame): training set
        test_df (pandas.DataFrame): test set

    Returns:
        train_df, test_df (tuple): standarized training and test set
    """
    # Calculate mean and standard deviation of the training set
    train_mean = train_df.mean()
    train_std = train_df.std()

    # Standarization of the data throughout the mean and standard deviation across train_df
    train_df = (train_df - train_mean) / train_std
    # val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, test_df
