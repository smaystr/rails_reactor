import numpy as np
import itertools
from copy import deepcopy


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    n = X.shape[0]

    if shuffle:
        if random_state:
            np.random.seed(random_state)

        p = np.random.permutation(n)
        X = X[p]
        y = y[p]

    ind = int(test_size * n)

    X_train, X_test = X[:-ind, :], X[-ind:, :]
    y_train, y_test = y[:-ind], y[-ind:]

    return X_train, X_test, y_train, y_test


class KFold:
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n = X.shape[0]

        if self.shuffle:
            if self.random_state:
                np.random.seed(random_state)
            p = np.random.permutation(n)
            X = X[p]
            y = y[p]

        indexes = np.arange(n)
        parts = np.array_split(indexes, self.n_splits)
        for i in range(self.n_splits):
            train = np.hstack(np.delete(parts, i))
            train_index, test_index = train, parts[i]

            yield train_index, test_index

    def get_n_splits(self):
        return self.n_splits


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=3):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        # self.

    def fit(self, X, y):
        res = {}
        kf = KFold(n_splits=self.cv)
        self.best_estimator = deepcopy(self.estimator)

        # print
        sorted_names = sorted(self.param_grid)
        # print(sorted_param)
        combinations = list(itertools.product(
            *(self.param_grid[name] for name in sorted_names)))

        print(f'Checking {len(combinations)} combinations...')

        for combination in combinations:
            # print(combination)
            for name, value in zip(sorted_names, combination):
                setattr(self.best_estimator, name, value)

            scores = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                self.best_estimator.fit(X_train, y_train)
                scores.append(self.best_estimator.score(X_test, y_test))

            res[tuple(combination)] = np.mean(scores)

        if self.best_estimator.best_is_max:
            best_params = max(res, key=res.get)
        else:
            best_params, best_score = min(res, key=res.get)

        for name, value in zip(sorted_names, best_params):
            setattr(self.best_estimator, name, value)
        self.best_estimator.fit(X_train, y_train)

    def predict(self, X):
        return self.best_estimator.predict(X)


class RandomizedSearchCV:

    def __init__(self, estimator, param_grid, cv=3, n_iters=10):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_iters = n_iters

    def fit(self, X, y):
        res = {}
        kf = KFold(n_splits=self.cv)
        self.best_estimator = deepcopy(self.estimator)

        sorted_names = sorted(self.param_grid)
        combinations = list(itertools.product(
            *(self.param_grid[name] for name in sorted_names)))

        print(f'Checking {self.n_iters} combinations...')

        indexes = np.random.choice(len(combinations), self.n_iters)
        for combination in  np.array(combinations)[indexes]:
            for name, value in zip(sorted_names, combination):
                setattr(self.best_estimator, name, value)

            scores = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                self.best_estimator.fit(X_train, y_train)
                scores.append(self.best_estimator.score(X_test, y_test))

            res[tuple(combination)] = np.mean(scores)

        if self.best_estimator.best_is_max:
            best_params = max(res, key=res.get)
        else:
            best_params, best_score = min(res, key=res.get)

        for name, value in zip(sorted_names, best_params):
            setattr(self.best_estimator, name, value)
        self.best_estimator.fit(X_train, y_train)

    def predict(self, X):
        return self.best_estimator.predict(X)