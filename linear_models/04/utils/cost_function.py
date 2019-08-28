import numpy as np
from typing import Callable


def cost_function(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    reg_param: float,
    decision_function: Callable,
) -> float:
    """
    Shows how accurate our model is based on current model parameters.

    Calculates using Ridge Regression method (L2 regularization)
    """
    num_examples = X.shape[0]
    predictions = decision_function(X, theta)
    loss = predictions - y

    # We should not regularize the parameter theta_zero.
    theta_cut = theta[1:, 0]

    # Calculate current predictions cost.
    # (np.sum(np.square(loss))) can be transformed to (loss @ loss.T)[0][0]
    cost = (1 / (2 * num_examples)) * (
        np.sum(np.square(loss)) + reg_param * np.sum(np.square(theta_cut))
    )
    return cost
