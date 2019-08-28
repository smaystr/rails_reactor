import numpy as np


class KFold:
    def __init__(self, folds=5, shuffle=False):
        if folds < 2:
            raise Exception(f"Must pass at least 2 folds. You passed {folds}.")

        self.folds = int(folds)
        self.shuffle = shuffle

    def split(self, X):
        indicies = np.arange(0, len(X))
        if self.shuffle:
            np.random.shuffle(indicies)

        validation_size = int(len(X) * 1 / self.folds)

        validation_folds = [
            indicies[indices * validation_size : (indices + 1) * validation_size]
            for indices in range(self.folds)
        ]

        return [(indicies, indicies[~i]) for i in validation_folds]


class TimeKFold:
    def __init__(self, folds=5):
        if folds < 2:
            raise Exception(f"Must pass at least 2 folds. You passed {folds}.")

        self.folds = int(folds)

    def split(self, X, time_col):
        indicies = np.arange(0, len(X))

        validation_size = int(len(X) * 1 / (self.folds + 1))

        validation_folds = [
            indicies[(indices + 1) * validation_size : (indices + 2) * validation_size]
            for indices in range(self.folds)
        ]

        return [(indicies[: i[0]], i) for i in validation_folds]


class LeaveOneOut:
    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def split(self, X):
        indices = np.arange(len(X))

        if self.shuffle:
            np.random.shuffle(indices)

        return [(np.delete(indices, i), [i]) for i in indices]


def train_test_split(X, y, test_size=0.25, shuffle=True):

    test_shape = int(test_size * X.shape[0])
    train_shape = X.shape[0] - test_shape
    if shuffle:
        train_indicies = np.random.choice(
            range(X.shape[0]), size=train_shape, replace=False
        )
    else:
        train_indicies = np.arange(train_shape)
    val_indicies = np.delete(np.arange(X.shape[0]), train_indicies)
    return (
        X[train_indicies],
        X[val_indicies],
        np.expand_dims(y[train_indicies], axis=1),
        np.expand_dims(y[val_indicies], axis=1),
    )
