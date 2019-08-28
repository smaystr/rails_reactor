import numpy
import model_selection
from abc import ABC, abstractmethod
from itertools import product


class Search(ABC):
    def __init__(self,
                 model,
                 parameter_grid,
                 split_type,
                 test_size=None,
                 n_splits=None,
                 time_column=None,
                 iterations=None,
                 metric=None
                 ):
        """
        General Search abstract class
        :type parameter_grid: dict
        :type split_type: int
        :type test_size: float
        :type n_splits: int
        :type time_column: str
        :type iterations: int
        :type metric: method
        """
        self.__validate_grid(model=model, parameter_grid=parameter_grid)
        self._test_size = test_size
        self._model_constructor = model.__class__
        self._parameter_grid = list(self.create_grid(parameter_grid, iterations))
        self._iterations = iterations
        self._n_splits = n_splits
        self._time_column = time_column
        self._split_type = split_type
        self._metric = metric
        self._best_score = None
        self._best_parameters = None

    @abstractmethod
    def create_grid(self, parameter_grid, iterations):
        """
        Creating the grid related to concrete object
        :type parameter_grid: dict
        :type iterations: int
        """
        pass

    def __validate_grid(self, model, parameter_grid):
        """
        Check if the class constructor contains the parameter grid hyperparameters
        :type parameter_grid: dict
        """
        for key in parameter_grid:
            if key not in model.__dict__:
                raise AttributeError(f"{key} is not present in {model} set of attributes")

    def fit(self, X_train, y_train):
        """
        Fit the X, y matrices and receive the model with best hyperparameters
        :type X_train: numpy.ndarray
        :type y_train: numpy.ndarray
        """
        scores = []
        for parameters in self._parameter_grid:
            model = self._model_constructor(**parameters)
            model.fit(X_train, y_train)
            score = model_selection.cross_validation_score(model, X_train, y_train, split_type=self._split_type, test_size=self._test_size, n_splits=self._n_splits, time_column=self._time_column, metric=self._metric)
            scores.append(score)
        scores = numpy.array(scores)
        best_score_arg = numpy.argmax(scores)
        self._best_score = scores[best_score_arg]
        self._best_parameters = self._parameter_grid[best_score_arg]
        return self._model_constructor(**self._best_parameters)

    def get_metric(self):
        return self._metric.__name__

    def get_best_parameters(self):
        return self._best_parameters

    def get_best_score(self):
        return self._best_score


class GridSearch(Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_grid(self, parameter_grid, iterations):
        """
        Fitting the dictionary of our parameters
        :type parameter_grid: dict
        :param iterations: int
        """
        keys = parameter_grid.keys()
        values = parameter_grid.values()
        for instance in product(*values):
            yield dict(zip(keys, instance))


class RandomSearch(Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_grid(self, parameter_grid, iterations):
        parameter_values = []
        for key, value in parameter_grid.items():
            if type(value) is list:
                values = numpy.random.choice(numpy.array(value), size=iterations)
            elif hasattr(value, 'rvs'):
                values = value.rvs(size=iterations)
            else:
                raise AttributeError(f"Unrecognized object for param generation {values}.\n\
                        Should be list or some random variable with rvs method implemented")
            parameter_values.append(values)
        parameter_values = numpy.array(parameter_values)
        keys = parameter_grid.keys()
        final_parameters = [{key: value for (key, value) in zip(keys, value)} for value in parameter_values.T]
        return final_parameters
