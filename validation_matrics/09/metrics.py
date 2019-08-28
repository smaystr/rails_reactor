import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)


def precision(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    return tp / (tp + fp)


def recall(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    return tp / (tp + fn)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
