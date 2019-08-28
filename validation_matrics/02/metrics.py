import numpy as np


def accuracy(y_pred, y_true):
    return np.mean(y_true == y_pred)


def precision(y_pred, y_true):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    return tp / (tp + fp)


def recall(y_pred, y_true):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return tp / (tp + fn)


def f1(y_pred, y_true):
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    return 2 * p * r / (p + r)


def mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))


def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))


def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))
