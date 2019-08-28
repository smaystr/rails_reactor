import torch
import time
from typing import List

from utils.gradient_descent import gradient_descent, stochastic_gradient_descent, mini_batch_sgd
from utils.dataset_processing import add_ones_column
from utils.metrics import get_metric


class BaseModel:
    def __init__(
        self,
        device: torch.device,
        *,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        C: float = 1,
    ):
        self.device = device
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_param = C
        self.theta = None
        self.cost_history = None
        self.fit_time_ = None

    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        start = time.time()
        if self.fit_intercept:
            X = add_ones_column(X, self.device)
        y = y.reshape(-1, 1)
        self.theta, self.cost_history = mini_batch_sgd(
            X,
            y,
            self.num_iterations,
            self.learning_rate,
            self.reg_param,
            self.device,
            self._decision_function,
        )
        self.fit_time_ = time.time() - start

    def predict(self, X: torch.tensor) -> torch.tensor:
        if self.fit_intercept:
            X = add_ones_column(X, self.device)
        predictions = self._decision_function(X, self.theta)
        return predictions.reshape((-1,))

    def score(self, X: torch.tensor, y: torch.tensor, metric_type: str) -> float:
        metric = get_metric(metric_type)
        y_pred = self.predict(X)
        return metric(y, y_pred)

    def get_thetas(self) -> torch.Tensor:
        return self.theta

    def get_cost_history(self) -> List:
        return self.cost_history

    @staticmethod
    def _decision_function(data: torch.tensor, theta: torch.tensor) -> torch.tensor:
        return torch.mm(data.float(), theta)


class BaseTorchModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        torch.nn.Module.__init__(self)
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, X):
        raise NotImplementedError
