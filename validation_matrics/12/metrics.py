import numpy as np


def mean_squared_error(y_real, y_pred):
    y_real = y_real.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    return np.mean(np.square(y_real - y_pred))


def accuracy_score(y_real, y_pred):
    y_real = y_real.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    return np.sum(y_real == y_pred) / y_real.shape[0]


def precision_score(y_real, y_pred):
    TP = np.sum(np.logical_and(y_pred == 1, y_real == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y_real == 0))

    return TP / (TP + FP)

def recall_score(y_real, y_pred):
    TP = np.sum(np.logical_and(y_pred == 1, y_real == 1))
    FN = np.sum(np.logical_and(y_pred == 0, y_real == 1))

    return TP / (TP + FN)

def f1_score(y_real, y_pred):
    prec = precision_score(y_real, y_pred)
    rec = recall_score(y_real, y_pred)

    return 2 * (prec * rec) / (prec + rec)

