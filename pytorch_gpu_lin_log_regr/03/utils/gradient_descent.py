import torch
from typing import Callable, Tuple


def _cost_function(
    X: torch.tensor,
    y: torch.tensor,
    theta: torch.tensor,
    reg_param: float,
    decision_function: Callable,
) -> float:
    """
    Calculates using Ridge Regression method (L2 regularization)
    """
    num_examples = X.shape[0]
    predictions = decision_function(X, theta)
    loss = predictions - y.float()
    # We should not regularize the parameter theta_zero.
    theta_cut = theta[1:, 0]
    # Calculate current predictions cost.
    cost = (1 / (2 * num_examples)) * (
        torch.sum(torch.mul(loss, loss)) + reg_param * torch.sum(torch.mul(theta_cut, theta_cut))
    )
    return cost


def _gradient_step(
    X: torch.tensor,
    y: torch.tensor,
    theta: torch.tensor,
    learning_rate: float,
    reg_param: float,
    decision_function: Callable
) -> torch.tensor:
    """
    Calculates what deltas should be taken for each parameter in
    order to minimize the cost function.
    """
    num_examples = X.shape[0]
    predictions = decision_function(X, theta)
    loss = predictions - y.float()
    theta = theta * (1 - learning_rate * reg_param / num_examples) - (learning_rate / num_examples) \
            * torch.mm(X.t().float(), loss)
    # We should NOT regularize the parameter theta_zero.
    theta[0] = theta[0] - learning_rate * (1 / num_examples) * torch.mm(X[:, 0].reshape((1, -1)).float(), loss)
    return theta


def gradient_descent(
    X: torch.tensor,
    y: torch.tensor,
    num_iterations: int,
    learning_rate: float,
    reg_param: float,
    device: torch.device,
    decision_function: Callable,
    random_seed: int = 0
) -> Tuple[torch.tensor, list]:
    torch.manual_seed(random_seed)
    num_features = X.shape[1]
    theta = torch.randn(num_features, 1, device=device)
    cost_history = []
    for _ in range(num_iterations):
        theta = _gradient_step(X, y, theta, learning_rate, reg_param, decision_function)
        cost_history.append(_cost_function(X, y, theta, reg_param, decision_function))
    return theta, cost_history


def mini_batch_sgd(
    X: torch.tensor,
    y: torch.tensor,
    num_iterations: int,
    learning_rate: float,
    reg_param: float,
    device: torch.device,
    decision_function: Callable,
    random_seed: int = 0,
    batch_size: int = 20
) -> Tuple[torch.tensor, list]:
    torch.manual_seed(random_seed)
    num_examples = X.shape[0]
    num_features = X.shape[1]
    theta = torch.randn(num_features, 1, device=device)
    cost_history = []

    for _ in range(num_iterations):
        for row_idx in range(0, num_examples, batch_size):
            if row_idx + batch_size >= num_examples:
                batch_size = num_examples - row_idx
            _X = X[row_idx:row_idx + batch_size, :].reshape(batch_size, -1)
            _y = y[row_idx:row_idx + batch_size].reshape(batch_size, -1)

            theta = _gradient_step(_X, _y, theta, learning_rate, reg_param, decision_function)
            cost_history.append(_cost_function(_X, _y, theta, reg_param, decision_function))
    return theta, cost_history


def stochastic_gradient_descent(
    X: torch.tensor,
    y: torch.tensor,
    num_iterations: int,
    learning_rate: float,
    reg_param: float,
    device: torch.device,
    decision_function: Callable,
    random_seed: int = 0
) -> Tuple[torch.tensor, list]:
    return mini_batch_sgd(X, y, num_iterations, learning_rate, reg_param, device, decision_function, random_seed, 1)
