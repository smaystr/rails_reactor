import numpy as np
import torch


def accuracy(y_, y):
    return torch.mean((y == y_).float())


def true_positive(y_, y):
    return torch.sum(y_ * y).float()


def false_positive(y_, y):
    return torch.sum(y_ * (1 - y)).float()


def true_negative(y_, y):
    return torch.sum((1 - y_) * (1 - y)).float()


def false_negative(y_, y):
    return torch.sum((1 - y_) * y).float()


def precision(y_, y):
    tp = true_positive(y_, y)
    fp = false_positive(y_, y)
    return tp / (tp + fp)


def recall(y_, y):
    tp = true_positive(y_, y)
    fn = false_negative(y_, y)
    return tp / (tp + fn)


def f_score(y_, y):
    p = precision(y_, y)
    r = recall(y_, y)
    return 2 * p * r / (p + r)


def mse(y_, y):
    return torch.mean((y_ - y).pow(2))


def rmse(y_, y):
    return torch.sqrt(mse(y_, y))


def mae(y_, y):
    return torch.mean((y_ - y).abs())


def confusion_matrix(y_, y, norm=False):
    tp = true_positive(y_, y)
    fp = false_positive(y_, y)
    tn = true_negative(y_, y)
    fn = false_negative(y_, y)

    matrix = np.asarray([[tp, fp], [fn, tn]])
    if norm:
        return matrix / (tp + fp + tn + fn)
    else:
        return matrix
