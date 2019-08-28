import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    prec = (tp + 1e-40) / (tp + fp + 1e-40)
    return prec


def recall(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    rec = (tp + 1e-40) / (tp + fn + 1e-40)
    return rec


def MSE(y_true, y_pred):
    return 1 / len(y_true) * np.sum(np.square(y_pred - y_true))


def MAE(y_true, y_pred):
    return 1 / len(y_true) * np.sum(np.abs(y_pred - y_true))


def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))
