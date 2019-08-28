import numpy as np


def prepare_data(x, y, fit_intercept):
    x = np.array(x)
    y = np.array(y)
    if fit_intercept:
        x_new = np.ones((x.shape[0], x.shape[1] + 1))
        x_new[:, 1:] = x
        x = x_new
    return x, y


def mse(y_true, y_pred):
    return np.sum((y_pred - y_true) ** 2) / y_true.size


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(y_true, y_pred):
    class1 = -y_true * np.log(y_pred)
    class2 = (1 - y_true) * np.log(1 - y_pred)
    loss = (class1 + class2).sum() / y_true.size
    return loss


def accuracy(y_true, y_pred):
    diff = y_pred - y_true
    return 1.0 - (float(np.count_nonzero(diff)) / diff.size)
