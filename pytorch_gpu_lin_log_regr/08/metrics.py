import torch


def accuracy(y_true, y_pred):
    return torch.mean((y_pred == y_true).float())


def true_positive(y_true, y_pred):
    return torch.sum(y_true * y_pred).float()


def false_positive(y_true, y_pred):
    return torch.sum(y_true * (1 - y_pred)).float()


def false_negative(y_true, y_pred):
    return torch.sum((1 - y_true) * y_pred).float()


def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp)


def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred).pow(2))


def rmse(y_true, y_pred):
    return torch.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return torch.mean((y_true - y_pred).abs())
