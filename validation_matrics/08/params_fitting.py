import itertools
import logging
from abc import ABC, abstractmethod

import numpy as np

from hw4.cross_validation import cross_val_score, KFold


class BaseSearch(ABC):
    def __init__(self, estimator, params, splits=5):
        self.estimator = estimator
        self.params = params
        self.splits = splits
        self.best_score = None
        self.best_params = {}
        self.best_weights = None

    def fit(self, X, y=None, split_class=KFold):
        params = self._create_grid(self.params)
        split_test_scores = []
        weights = []
        for i in range(len(params)):
            logging.info(f'Fitting params {i + 1} out of {len(params)}')
            res = cross_val_score(self.estimator(C=params[i]['C'], num_iterations=int(params[i]['num_iterations']),
                                                 learning_rate=params[i]['learning_rate']), X, y, splits=self.splits,
                                  split_class=split_class)
            split_test_scores.append(res['scoring']['test'])
            weights.append(res['weights'])

        mean_scores = []
        for i in range(len(split_test_scores)):
            mean_scores.append(split_test_scores[i].mean())

        self.best_score = np.array(mean_scores).max()
        self.best_params = params[mean_scores.index(self.best_score)]
        self.best_weights = weights[mean_scores.index(self.best_score)]
        return self

    @staticmethod
    def cartesian(parameters):
        keys, values = zip(*parameters.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    @abstractmethod
    def _create_grid(self, parameters):
        raise NotImplementedError


class GridSearch(BaseSearch):

    def __init__(self, estimator, params, splits=5):
        super().__init__(estimator, params, splits)

    def _create_grid(self, parameters):
        return self.cartesian(parameters)


class RandSearch(BaseSearch):

    def __init__(self, estimator, param_distributions, splits=5, n_iters=100, random_state=42):
        super().__init__(estimator, param_distributions, splits)
        self.n_iters = n_iters
        self.random_state = random_state

    def _create_grid(self, parameters):
        n = np.int(np.floor(self.n_iters / len(parameters)))
        np.random.seed(self.random_state)

        return self.cartesian({prop: np.random.uniform(*parameters[prop], n).tolist() for prop in parameters})
