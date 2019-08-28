import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def recall(y_true, y_pred):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    print(f'tp = {tp}, fn = {fn}')
    return tp / (tp + fn)


def precision(y_true, y_pred):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    print(f'tp = {tp}, fp = {fp}')
    return tp / (tp + fp)


def F1(y_true, y_pred):
    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)
    return 2 * rec * prec / (rec + prec)
