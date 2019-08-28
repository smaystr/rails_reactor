import numpy as np


class LabelEncoder:
    def __init__(self, starting_val=0):
        self.starting_val = starting_val

    def fit(self, data, data_cols):
        # pass in string columns
        if data.shape[1] != len(data_cols):
            raise Exception("Number of columns doesn't correspond to data shape.")

        self.unique = {i: {} for i in data_cols}
        for key, column in enumerate(data_cols):
            unique_col = np.unique(data[:, key])
            for key2, i in enumerate(unique_col):
                self.unique[column][i] = key2 + self.starting_val

    def transform(self, data, data_cols):

        if not set(data_cols).issubset(set(self.unique.keys())):
            raise Exception("New columns encountered")

        for i in data_cols:
            for key, val in self.unique[i].items():
                data[data == key] = val
        return data.astype(float)

    def fit_transform(self, data, data_cols):
        self.fit(data, data_cols)
        return self.transform(data, data_cols)


class StandardScaler:
    def __init__(self, with_std=True, with_mean=True):
        self.with_std = with_std
        self.with_mean = with_mean
        self.mean = 0
        self.std = 1

    def fit(self, X):
        if self.with_std:
            self.std = X.std(axis=0)
        if self.with_mean:
            self.mean = X.mean(axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
