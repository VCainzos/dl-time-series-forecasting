def split(df, train=0.9, test=0.1):
    n = len(df)
    train_df = df[0 : int(n * 0.9)]
    # val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n * 0.9) :]

    num_features = df.shape[1]
    return train_df, test_df


def standarize(train_df, test_df):
    # Calculate mean and standard deviation of the training set
    train_mean = train_df.mean()
    train_std = train_df.std()

    # Standarization of the data throughout the mean and standard deviation across train_df
    train_df = (train_df - train_mean) / train_std
    # val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, test_df
