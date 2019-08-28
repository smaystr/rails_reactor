import numpy as np

# regression metrics
def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / y_true.shape[0]

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / y_true.shape[0]

def mape(y_true, y_pred):
    return 100 * np.abs((y_true - y_pred) / y_true).sum() / y_true.shape[0]

def mpe(y_true, y_pred):
    return 100 * ((y_true - y_pred) / y_true).sum() / y_true.shape[0]

def r2(y_true, y_pred):
    return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

# classification metrics
def accuracy(y_true, y_pred):
     return (y_true == y_pred).sum() / y_true.shape[0]

def recall(y_true, y_pred):
     return (np.int64(y_true) & np.int64(y_pred)).sum() / np.int64(y_true).sum()

def precision(y_true, y_pred):
     return (np.int64(y_true) & np.int64(y_pred)).sum() / np.int64(y_pred).sum()

def fbeta(y_true, y_pred, beta):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return (1 + beta ** 2) * (prec * rec) / ((beta ** 2) * prec + rec)

def f1(y_true, y_pred):
    return fbeta(y_true, y_pred, 1)

def log_loss(y_true, y_pred, eps=1e-15):
     return - (y_true.ravel() * np.log(y_pred[:, 1]) + (1 - y_true.ravel()) * np.log(1 - y_pred[:, 1])).sum() / y_true.shape[0]
