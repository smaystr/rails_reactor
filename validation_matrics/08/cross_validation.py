import time

import numpy as np


def train_test_split(x, y, test_size=0.2, random_state=42):
    test = np.round(x.shape[0] * test_size)

    np.random.seed(random_state)
    test_indices = np.random.choice(np.arange(x.shape[0]), size=np.int(test), replace=False)

    x_train = np.zeros((0, x.shape[1]))
    x_test = np.zeros((0, x.shape[1]))
    y_train = np.zeros((0, y.shape[1]))
    y_test = np.zeros((0, y.shape[1]))

    for i in range(x.shape[0]):
        if np.any(test_indices == i):
            x_test = np.vstack([x_test, x[i, :]])
            y_test = np.vstack([y_test, y[i, :]])
        else:
            x_train = np.vstack([x_train, x[i, :]])
            y_train = np.vstack([y_train, y[i, :]])

    return x_train, x_test, y_train, y_test


def feature_importance(weights, train_columns):
    # exclude bias
    if train_columns is None:
        return []
    weights = np.abs(weights[1:].flatten())
    args = np.argsort(weights)[::-1]
    return list(zip(weights[args], train_columns[args]))


class BaseCrossValidator:

    def __init__(self, n_splits=5, shuffle=False, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x, y=None, _groups=None):
        fold_size = x.shape[0] / self.get_n_splits(x=x, y=y)
        if np.mod(fold_size, 1) == 0:
            fold_size = np.int(fold_size)
        else:
            fold_size = np.int(np.floor(fold_size))

        np.random.seed(self.random_state)
        if self.shuffle:
            indices = np.random.choice(np.arange(x.shape[0]),
                                       size=np.int(x.shape[0]),
                                       replace=False)
        else:
            indices = np.arange(x.shape[0])

        index = 0

        for i in range(self.get_n_splits(x)):
            test = indices[(indices >= index) & (indices < index + fold_size)]
            train = indices[(indices < index) | (indices >= index + fold_size)]
            index += fold_size

            yield train, test

    def get_n_splits(self, x=None, y=None):
        raise NotImplementedError


class KFold(BaseCrossValidator):
    def get_n_splits(self, x=None, y=None, _groups=None):
        return self.n_splits


class LeaveOneOut(BaseCrossValidator):
    def get_n_splits(self, x=None, y=None):
        return x.shape[0]


def cross_val_score(estimator, x, y=None, split_class=KFold, splits=5, shuffle=False, random_state=42, columns=None):
    sc = split_class(splits, shuffle, random_state)

    test_score = np.array([])
    train_score = np.array([])
    fit_time = np.array([])
    score_time = np.array([])
    for train, test in sc.split(x, y):
        curr = time.time()
        estimator.fit(x[train], y[train])
        fit_time = np.append(fit_time, time.time() - curr)
        curr = time.time()
        test_score = np.append(test_score, estimator.score(x[test], y[test]))
        score_time = np.append(score_time, time.time() - curr)

        train_score = np.append(train_score, estimator.score(x[train], y[train]))
    res = {
        'scoring': {
            'test': test_score,
            'train': train_score
        },
        'timing': {
            'fit': fit_time,
            'score': score_time
        },
        'weights': estimator.weights,
        'feature_importance': feature_importance(estimator.weights, columns)
    }

    return res
