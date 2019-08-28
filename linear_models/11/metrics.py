import numpy as np


def mean_squared_error(y_real, y_pred):
    y_real = y_real.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    return np.mean(np.square(y_real - y_pred))


def accuracy_score(y_real, y_pred):
    y_real = y_real.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    return np.sum(y_real == y_pred) / y_real.shape[0]
