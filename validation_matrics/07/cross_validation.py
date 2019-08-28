import numpy as np
import time
from abc import ABC, abstractmethod


def train_test_split(X, y, test_size=0.1, random_state=0):
    test = np.round(X.shape[0] * test_size)

    np.random.seed(random_state)
    test_indices = np.random.choice(np.arange(X.shape[0]), size=np.int(test), replace=False)

    X_train = np.zeros((0, X.shape[1]))
    X_test = np.zeros((0, X.shape[1]))
    y_train = np.zeros((0, y.shape[1]))
    y_test = np.zeros((0, y.shape[1]))

    for i in range(X.shape[0]):
        if np.any(test_indices == i):
            X_test = np.vstack([X_test, X[i, :]])
            y_test = np.vstack([y_test, y[i, :]])
        else:
            X_train = np.vstack([X_train, X[i, :]])
            y_train = np.vstack([y_train, y[i, :]])
    
    return X_train, X_test, y_train, y_test


class BaseCrossValidator(ABC):

    def __init__(self, n_splits=5, shuffle=False, random_state=0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        fold_size = X.shape[0] / self.get_n_splits(X)
        fold_size = np.int(fold_size) if np.mod(fold_size, 1) == 0 else np.int(np.floor(fold_size))

        np.random.seed(self.random_state)
        indices = np.random.choice(np.arange(X.shape[0]), size=np.int(X.shape[0]), replace=False) if self.shuffle else np.arange(X.shape[0])
        test_index = 0

        for i in range(self.get_n_splits(X)):
            test = indices[(indices >= test_index) & (indices < test_index + fold_size)]
            train = indices[(indices < test_index) | (indices >= test_index + fold_size)]
            test_index += fold_size

            yield train, test
    
    @abstractmethod
    def get_n_splits(self):
        raise NotImplementedError


class KFold(BaseCrossValidator):

    def __init__(self, n_splits=5, shuffle=False, random_state=0):
        super().__init__(n_splits, shuffle, random_state)
    
    def get_n_splits(self, X=None, y=None):
        return self.n_splits


class LeaveOneOut(BaseCrossValidator):

    def __init__(self):
        super().__init__()
    
    def get_n_splits(self, X, y=None):
        return X.shape[0]


def cross_validate(estimator, X, y=None, cv=5, return_train_score=False, return_estimator=False):
    test_score = np.array([])
    train_score = np.array([])
    fit_time = np.array([])
    score_time = np.array([])
    estimators = []

    for train, test in KFold(cv).split(X, y):
        curr = time.time()
        est = estimator.fit(X[train], y[train])
        fit_time = np.append(fit_time, time.time() - curr)
        curr = time.time()
        test_score = np.append(test_score, estimator.score(X[test], y[test]))
        score_time = np.append(score_time, time.time() - curr)
        
        if return_train_score:
            train_score = np.append(train_score, estimator.score(X[train], y[train]))

        if return_estimator:
            estimators.append(est)

    res = {'test_score': test_score,
            'train_score': train_score,
            'fit_time': fit_time,
            'weights': estimator.coef}

    if return_train_score:
        res.update({'score_time': score_time})
    if return_estimator:
        res.update({'estimator': estimators})
    
    return res
