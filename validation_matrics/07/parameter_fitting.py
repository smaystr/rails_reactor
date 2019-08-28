import numpy as np
from abc import ABC, abstractmethod
from cross_validation import cross_validate


class BaseSearch(ABC):
    def __init__(self, estimator, param_grid, cv=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.cv_results = {}
        self.best_score = None
        self.best_hyperparameters = {}
    
    def fit(self, X, y=None):
        params = self._create_grid(self.param_grid)

        mean_fit_time = np.zeros(len(params))
        std_fit_time = np.zeros(len(params))
        mean_score_time = np.zeros(len(params))
        std_score_time = np.zeros(len(params))
        mean_test_score = np.zeros(len(params))
        std_test_score = np.zeros(len(params))
        rank_test_score = np.zeros(len(params))
        split_test_scores = []
        weights = []

        for i in range(len(params)):
            res = cross_validate(self.estimator(**params[i]),X, y, cv=self.cv, return_train_score=True)

            mean_fit_time[i] = res['fit_time'].mean()
            std_fit_time[i] = res['fit_time'].std()
            mean_score_time[i] = res['score_time'].mean()
            std_score_time[i] = res['score_time'].std()
            mean_test_score[i] = res['test_score'].mean()
            std_test_score[i] = res['test_score'].std()
            split_test_scores.append(res['test_score'])
            weights.append(res['weights'])
        
        self.cv_results.update({ 'mean_fit_time': mean_fit_time,
                                'std_fit_time': std_fit_time,
                                'mean_score_time': mean_score_time,
                                'std_score_time': std_score_time,
                                'mean_test_score': mean_test_score,
                                'std_test_score': std_test_score,
                                'params': params
                                })

        mean_scores = []
        for i in range(len(split_test_scores)):
            self.cv_results.update({f'split{i}_test_score': split_test_scores[i]})
            mean_scores.append(split_test_scores[i].mean())

        self.best_score = np.array(mean_scores).max()
        self.best_hyperparameters = params[mean_scores.index(self.best_score)]
        self.best_weights = weights[mean_scores.index(self.best_score)]

        return self

    def cartesian(self, parameters):
        combinations = np.array(np.meshgrid(*list(parameters.values()))).T.reshape(-1, len(parameters))
        params = []
        for i in range(len(combinations)):
            params.append(dict(zip(parameters.keys(),combinations[i])))

        return params
    
    @abstractmethod
    def _create_grid(self, parameters):
        raise NotImplementedError


class GridSearch(BaseSearch):
    
    def __init__(self, estimator, param_grid, cv=5):
        super().__init__(estimator, param_grid, cv)
    
    def _create_grid(self, parameters):
        return self.cartesian(parameters)


class RandomizedSearch(BaseSearch):

    def __init__(self, estimator, param_distributions, cv=5, n_iters=10, random_state=0):
        super().__init__(estimator, param_distributions, cv=5)
        self.n_iters = n_iters
        self.random_state = random_state

    def _create_grid(self, parameters):
        n = np.int(np.floor(self.n_iters / len(parameters)))
        param_grid = {}

        np.random.seed(self.random_state)
        for prop in parameters:
            param_grid.update({prop: np.random.uniform(*parameters[prop], n).tolist()})

        return self.cartesian(param_grid)
