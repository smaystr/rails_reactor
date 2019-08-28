import numpy as np
from typing import Callable, Tuple

from utils.cost_function import cost_function


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    num_iterations: int,
    learning_rate: float,
    reg_param: float,
    decision_function: Callable,
) -> Tuple[np.ndarray, list]:
    """
    Calculates what deltas should be taken for each parameter in
    order to minimize the cost function.
    """
    num_examples = X.shape[0]
    num_features = X.shape[1]
    theta = np.random.rand(num_features, 1)
    cost_history = []

    for _ in range(num_iterations):
        predictions = decision_function(X, theta)
        loss = predictions - y
        theta = theta * (1 - learning_rate * reg_param / num_examples) - (
            learning_rate / num_examples
        ) * (X.T @ loss)

        # We should NOT regularize the parameter theta_zero.
        theta[0] = theta[0] - learning_rate * (1 / num_examples) * (X[:, 0].T @ loss)
        cost_history.append(cost_function(X, y, theta, reg_param, decision_function))

    return theta, cost_history
