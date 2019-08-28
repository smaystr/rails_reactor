import numpy as np


class TrainTestCV:
    def __init__(self, X, test_size=0.2, shuffle=True):
        self.X = X
        self.test_size = test_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(self.X.shape[0])
        if self.shuffle:
            np.random.shuffle(indices)

        train_ind = indices[:self.X.shape[0] * self.test_size]
        test_ind = indices[self.X.shape * self.test_size:]
        yield train_ind, test_ind


class KFoldCV:
    def __init__(self, X, n_splits, shuffle=True):
        self.X = X
        self.n_splits = n_splits
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(self.X.shape[0])
        if self.shuffle:
            np.random.shuffle(indices)
        splits = np.array_split(indices, self.n_splits)

        for fold in range(0, self.n_splits):
            test_ind = splits[fold]
            train_ind = np.concatenate(
                [x for i, x in enumerate(splits) if i != fold]
            )
            yield train_ind, test_ind

    def __len__(self):
        return self.n_splits


class TimeTrainTestCV:
    def __init__(self, dates, test_size=0.2):
        self.dates = dates
        self.test_size = test_size

    def __iter__(self):
        indices = np.asarray(self.dates.argsort())

        values = int(self.dates.shape[0] * self.test_size)
        train_ind = indices[values:]
        test_ind = indices[:values]
        yield train_ind, test_ind


class LeaveOneOutCV(KFoldCV):
    def __init__(self, X, shuffle, n_splits):
        super().__init__(X, n_splits, shuffle)
        self.X = X
        self.n_splits = X.shape[0]
        self.shuffle = shuffle


class TimeSeriesCV:

    def __init__(self, dates, n_splits):
        self.dates = dates
        self.n_splits = n_splits

    def __iter__(self):
        indices = np.asarray(self.dates.argsort())
        split = np.array_split(indices, self.n_splits + 1)

        for i in range(self.n_splits):
            train_ind = split[i]
            test_ind = split[i + 1]

            yield train_ind, test_ind

    def __len__(self):
        return self.n_splits
