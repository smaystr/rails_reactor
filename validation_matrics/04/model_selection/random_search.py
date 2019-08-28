import numpy as np
from typing import Callable, List, Dict

from model_selection.grid_search import GridSearch


class RandomizedSearch(GridSearch):
    def __init__(
        self,
        estimator,
        param_grid: Dict,
        n_iter: int = 10,
        scoring: Callable = None,
        cv: Callable = None,
        refit: bool = True,
    ):
        self.n_iter = n_iter
        GridSearch.__init__(self, estimator, param_grid, scoring, cv, refit)

    def make_param_grid(self, params: Dict) -> List[Dict]:
        param_grid = GridSearch.make_param_grid(self, params)
        assert self.n_iter < len(param_grid)
        param_grid = np.random.choice(param_grid, self.n_iter).tolist()
        return param_grid
