import time
import itertools
import numpy as np
from typing import Callable, Union, List, Tuple, Dict


class GridSearch:
    def __init__(
        self,
        estimator,
        param_grid: Dict,
        scoring: Callable = None,
        cv: Callable = None,
        refit: bool = True,
    ):
        self.estimator = estimator
        self.param_grid = self.make_param_grid(param_grid)
        self.scoring = scoring
        self.cv = cv
        self.refit = refit

        self.cv_results_ = {}
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_index_ = None
        self.n_splits_ = None
        self.refit_time_ = None  # if refit = True
        self.search_time_ = None

    def fit(
        self,
        X: np.ndarray,
        y: Union[None, np.ndarray],
        groups: Union[np.ndarray, List, Tuple, None] = None,
    ) -> None:
        search_start = time.time()

        scores = []
        cross_validation = self.cv()
        for params_index, params in enumerate(self.param_grid):
            intermediate_scores = []
            for train_index, test_index in cross_validation.split(X, groups):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model = self.estimator(**params)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                intermediate_scores.append(score)
            scores.append(np.mean(intermediate_scores))
        self.best_score_ = self.scoring(scores)
        index = scores.index(self.best_score_)
        self.best_params_ = self.param_grid[index]

        if self.refit:
            refit_start = time.time()
            model = self.estimator(**self.best_params_)
            model.fit(X, y)
            self.refit_time_ = time.time() - refit_start
            self.best_estimator_ = model

        self.search_time_ = time.time() - search_start

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.refit:
            raise Exception("Only available if refit=True")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.refit:
            raise Exception("Only available if refit=True")
        return self.best_estimator_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.refit:
            raise Exception("Only available if refit=True")
        return self.best_estimator_.score(X, y)

    def get_param_grid(self):
        return self.param_grid

    def make_param_grid(self, params: Dict) -> List[Dict]:
        params_names = params.keys()
        params = params.values()
        params_combinations = list(itertools.product(*params))
        # self.param_grid = []
        # for combination in params_combinations:
        #     self.param_grid.append({name: value for name, value in zip(params_names, combination)})
        param_grid = [
            {name: value for name, value in zip(params_names, combination)} for combination in params_combinations
        ]
        return param_grid
