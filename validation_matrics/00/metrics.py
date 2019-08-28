import numpy as np


def mae(y_true, y_pred):
    return np.sum(abs(y_true - y_pred)) / y_true.size


def mse(y_true, y_pred):
    return np.sum((y_pred - y_true) ** 2) / y_true.size


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred):
    return 100 * np.sum(abs((y_true - y_pred) / y_true)) / y_true.size


def mpe(y_true, y_pred):
    return 100 * np.sum((y_true - y_pred) / y_true) / y_true.size


def log_loss(y_true, y_pred):
    class1 = -y_true * np.log(y_pred)
    class2 = (1 - y_true) * np.log(1 - y_pred)
    loss = (class1 + class2).sum() / y_true.size
    return loss


def accuracy(y_true, y_pred):
    diff = y_pred - y_true
    return 1.0 - (float(np.count_nonzero(diff)) / diff.size)


def precision(y_true, y_pred):
    y_true = np.array(y_true, dtype='int32')
    y_pred = np.array(y_pred, dtype='int32')
    tp = (y_true & y_pred).sum()
    return tp / y_pred.sum()


def recall(y_true, y_pred):
    y_true = np.array(y_true, dtype='int32')
    y_pred = np.array(y_pred, dtype='int32')
    tp = (y_true & y_pred).sum()
    return tp / y_true.sum()


def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec)
