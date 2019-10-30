import numpy as np
from typing import List

from utils.gradient_descent import gradient_descent
from utils.regression_metrics import get_metric
from utils.dataset_processing import add_ones_column

np.random.seed(42)


class LinearReg:
    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        C: float = 1,
    ):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_param = C
        self.theta = None
        self.cost_history = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.fit_intercept:
            X = add_ones_column(X)
        y = y.reshape(-1, 1)
        self.theta, self.cost_history = gradient_descent(
            X,
            y,
            self.num_iterations,
            self.learning_rate,
            self.reg_param,
            self._decision_function,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = add_ones_column(X)
        predictions = self._decision_function(X, self.theta)
        return predictions.reshape((-1,))

    def score(self, X: np.ndarray, y: np.ndarray, metric_type: str = "mse") -> float:
        metric = get_metric(metric_type)
        y_pred = self.predict(X)
        return metric(y, y_pred)

    def get_cost_history(self) -> List:
        return self.cost_history

    @staticmethod
    def _decision_function(data: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return LinearReg._hypothesis(data, theta)

    @staticmethod
    def _hypothesis(data: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return data @ theta