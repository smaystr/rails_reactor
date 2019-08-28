from __future__ import annotations
from functools import partial, reduce
from itertools import product
import operator
from copy import deepcopy
import numpy as np


class ParameterGrid:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        items = sorted(self.param_grid.items())
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params

    def __len__(self):
        product_partial = partial(reduce, operator.mul)
        return sum(product_partial(len(v) for v in self.param_grid.values()))


class ParameterSampler:
    def __init__(self, param_grid, n_iters):
        self.param_grid = param_grid
        self.n_iters = n_iters

    def __iter__(self):
        items = sorted(self.param_grid.items())
        for _ in range(self.n_iters):
            params = dict()
            for k, v in items:
                params[k] = np.random.choice(v, 1, False)[0]
            yield params

    def __len__(self):
        return self.n_iters


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv, scoring=None):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        try:
            self.metric = estimator.metrics[scoring]
        except KeyError:
            default = estimator.keys()[0]
            print(f'Metric {scoring} is not available, using {default}')
            self.metric = estimator.metrics[default]
        self.best_estimator_ = None
        self.best_score_ = None
        self.param_grid = ParameterGrid(param_grid)

    def _from_params(self, params):
        keys = params.keys()
        new_estimator = deepcopy(self.estimator)
        if "lr" in keys:
            new_estimator.lr = params["lr"]
        if "epoch" in keys:
            new_estimator.epoch = params["epoch"]
        if "penalty" in keys:
            new_estimator.penalty = params["penalty"]
        if "C" in keys:
            new_estimator.C = params["C"]
        return new_estimator

    def _retrain_full_data(self, X, y, params):
        estimator = self._from_params(params)
        estimator.fit(X, y)
        return estimator

    def fit(self, X, y):
        for params in self.param_grid:
            scores = []
            for train_index, test_index in self.cv:
                new_estimator = self._from_params(params)
                new_estimator.fit(X[train_index], y[train_index].reshape((-1, 1))
                                  )
                predictions = new_estimator.predict(X[test_index])
                scores.append(self.metric(predictions.reshape((-1, 1)), y[test_index].reshape((-1, 1))))

            mean_score = sum(scores) / len(scores)
            if self.best_score_:
                if self.best_score_ < mean_score:
                    self.best_score_ = mean_score
                    self.best_params_ = params
            else:
                self.best_score_ = mean_score
                self.best_params_ = params
        self.best_estimator_ = self._retrain_full_data(X, y, self.best_params_)
        return self

    def predict(self, X, y):
        if self.best_estimator_:
            return self.best_estimator_.predict(X, y)
        else:
            print("Models were not fitted yet, call this method after fit(X, y)")

    def score(self, X, y):
        if self.best_estimator_:
            return self.best_estimator_.score(X, y, self.scoring)
        else:
            print("Models were not fitted yet, call this method after fit(X, y)")


class RandomSearchCV(GridSearchCV):
    def __init__(self, estimator, param_grid, cv, scoring=None, n_iter=10):
        GridSearchCV.__init__(self, estimator, param_grid, cv, scoring)
        self.param_grid = ParameterSampler(param_grid, n_iter)
