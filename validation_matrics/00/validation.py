import numpy as np


def train_test_split(X, y=None, test_size=0.2, shuffle=True):
    X_copy = np.array(X, copy=True)
    if shuffle:
        np.random.shuffle(X_copy)
    index = int(np.ceil(X_copy.shape[0] * test_size))
    X_train = X_copy[index:, :]
    X_test = X_copy[: index, :]
    if y is not None:
        y_copy = np.array(y, copy=True)
        if shuffle:
            np.random.shuffle(y_copy)
        y_train = y_copy[index:]
        y_test = y_copy[: index]
        return X_train, X_test, y_train, y_test
    return X_train, X_test


class KFold:

    def __init__(self, n_splits=5, shuffle=False):
        self.n_splits = n_splits
        self.shuffle = shuffle

    def split(self, X):
        indices = np.arange(X.shape[0])
        n_samples = indices.size
        for i in range(self.n_splits):
            if i < n_samples % self.n_splits:
                size = n_samples // self.n_splits + 1
                train = np.delete(indices, np.s_[i * size: (i + 1) * size])
                test = indices[i * size: (i + 1) * size]
            else:
                size = n_samples // self.n_splits
                train = np.delete(indices, np.s_[i * size: (i + 1) * size])
                test = indices[i * size: (i + 1) * size]
            yield train, test


class LeaveOneOut(KFold):

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def split(self, X):
        self.n_splits = X.shape[0]
        return KFold.split(self, X)
