from itertools import combinations, product
from multiprocessing import Pool
import numpy as np


class GridSearch:
    def __init__(self, estimator, cv, param_distr, scorer, iters=5, jobs=1):
        self.estimator = estimator
        self.param_distr = param_distr
        self.jobs = jobs
        self.scorer = scorer
        self.iters = iters
        self.cv = cv

    def set_model(self, params):
        for i, key in enumerate(self.param_distr.keys()):
            try:
                setattr(self.estimator, key, params[i])
            except:
                print(f"Attribute {key} not present in model")

    def evaluate(self, params):

        history = np.zeros((self.cv.folds, 2))
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X)):
            train_X, test_X = self.X[train_idx], self.X[val_idx]
            train_y, test_y = self.y[train_idx], self.y[val_idx]
            self.set_model(params)
            self.estimator.fit(train_X, train_y)
            history[fold] = (
                self.scorer(train_y, self.estimator.predict(train_X)),
                self.scorer(test_y, self.estimator.predict(test_X)),
            )

        return [
            history[:, 0].mean(),
            history[:, 1].mean(),
            dict(zip(self.param_distr.keys(), params)),
        ]

    def fit(self, X, y):
        self.X = X
        self.y = y

        combs = np.array(list((product(*list(self.param_distr.values())))))

        with Pool(self.jobs) as pool:

            res = pool.map(self.evaluate, combs)

        best_fit = res[np.argmin(np.array([i[1] for i in res]))]

        self.train_score = best_fit[0]
        self.test_score = best_fit[1]
        self.best_params = best_fit[2]

        return {
            "train_score": best_fit[0],
            "test_score": best_fit[1],
            "params": best_fit[2],
        }


class GridSearch:
    def __init__(
        self, estimator, cv, param_distr, scorer, iters=5, task="regression", jobs=1
    ):
        self.estimator = estimator
        self.param_distr = param_distr
        self.jobs = jobs
        self.scorer = scorer
        self.iters = iters
        self.cv = cv
        self.task = task

    def set_model(self, params):
        for i, key in enumerate(self.param_distr.keys()):
            try:
                setattr(self.estimator, key, params[i])
            except:
                print(f"Attribute {key} not present in model")

    def evaluate(self, params):

        history = np.zeros((self.cv.folds, 2))
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X)):
            train_X, test_X = self.X[train_idx], self.X[val_idx]
            train_y, test_y = self.y[train_idx], self.y[val_idx]
            self.set_model(params)
            self.estimator.fit(train_X, train_y)
            history[fold] = (
                self.scorer(train_y, self.estimator.predict(train_X)),
                self.scorer(test_y, self.estimator.predict(test_X)),
            )

        return [
            history[:, 0].mean(),
            history[:, 1].mean(),
            dict(zip(self.param_distr.keys(), params)),
        ]

    def fit(self, X, y):
        self.X = X
        self.y = y

        combs = np.array(list((product(*list(self.param_distr.values())))))
        random_combs = combs[
            np.random.choice(np.arange(0, len(combs)), self.iters, replace=False)
        ]
        with Pool(self.jobs) as pool:

            res = pool.map(self.evaluate, random_combs)
        if self.task == "regression":
            best_fit = res[np.argmin(np.array([i[1] for i in res]))]
        else:
            best_fit = res[np.argmax(np.array([i[1] for i in res]))]

        self.train_score = best_fit[0]
        self.test_score = best_fit[1]
        self.best_params = best_fit[2]

        return {
            "train_score": best_fit[0],
            "test_score": best_fit[1],
            "params": best_fit[2],
        }
