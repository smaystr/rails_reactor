import torch


def accuracy(y_true, y_pred):
    return torch.mean((y_true == y_pred).float())


def recall(y_true, y_pred):
    tp = torch.sum(y_pred * y_true).float()
    fn = torch.sum((1 - y_pred) * y_true).float()
    return tp / (tp + fn)


def precision(y_true, y_pred):
    tp = torch.sum(y_pred * y_true).float()
    fp = torch.sum(y_pred * (1 - y_true)).float()
    return tp / (tp + fp)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def mse(y_true, y_pred):
    return torch.mean(torch.pow(y_pred - y_true, 2))


def rmse(y_true, y_pred):
    return torch.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true))
