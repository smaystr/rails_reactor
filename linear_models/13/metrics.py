import numpy as np

def accuracy(y_: np.ndarray, y: np.ndarray) -> float:
    return np.mean(y == y_)

def precision(y_: np.ndarray, y: np.ndarray) -> float:
    tp = np.sum(np.logical_and(y_ == 1, y == 1))
    fp = np.sum(np.logical_and(y_ == 1, y == 0))
    return tp / (tp + fp)

def recall(y_: np.ndarray, y: np.ndarray) -> float:
    tp = np.sum(np.logical_and(y_ == 1, y == 1))
    fn = np.sum(np.logical_and(y_ == 0, y == 1))
    return tp / (tp + fn)

def f_score(y_: np.ndarray, y: np.ndarray) -> float:
    p = precision(y_, y)
    r = recall(y_, y)
    return 2 * p * r / (p + r)

def mse(y_: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.square(y_ - y))

def rmse(y_: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(mse(y_, y))

def mae(y_: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.abs(y_ - y))
