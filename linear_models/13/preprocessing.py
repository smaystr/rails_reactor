from pathlib import Path
from typing import Union, Tuple
import numpy as np

def read_data(path: Union[str, Path], X_Y: bool = True, dtype: np.dtype = np.float32) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=dtype)
    if X_Y:
        return data[:, :-1], data[:, -1]
    return data

def read_feature_names(path: Path, skip_last: bool = True):
    with path.open(encoding='utf-8') as f:
        features = f.readline()
    features = features.strip().split(',')
    if skip_last:
        return features[:-1]
    return features

def onehot(data: np.ndarray, column: int, feature: list, params: list = []) -> Tuple[np.ndarray, int, list, list]:
    column_copy = data[:, column].astype(np.int32)

    if not params:
        unique = np.unique(column_copy)
    else:
        unique = params

    hotted = np.zeros((column_copy.size, unique.shape[0]))
    hotted[np.arange(column_copy.size), column_copy - unique.min()] = 1

    result = np.zeros(shape=(data.shape[0], data.shape[1] + hotted.shape[1] - 1))
    result[:, :column] = data[:, :column]
    result[:, column:column + hotted.shape[1]] = hotted
    result[:, column + hotted.shape[1]:] = data[:, column+1:]
    return result, unique.shape[0] - 1, unique, [f'{feature}_{un}' for un in unique]

def to_numeric(data: np.ndarray, column: int, classes: dict) -> np.ndarray:
    result = np.copy(data)
    for k, v in classes.items():
        result[:, column][result[:, column] == k] = v
    return result

def to_numeric_multiple(data: np.ndarray, columns: list) -> Tuple[np.ndarray, list]:
    classes_list = []
    result = np.copy(data)
    for c in columns:
        unique = np.unique(data[:, c])
        classes = {k:v for k, v in zip(unique, range(unique.shape[0]))}
        classes_list.append(classes)
        for k, v in classes.items():
            result[:, c][result[:, c] == k] = v
    return result, classes_list

def onehot_columns(data: np.ndarray, columns: list, features: list, params: list = []) -> Tuple[np.ndarray, list]:
    result = np.copy(data)
    computed = []
    move = 0

    if not params:
        params = [[] for _ in range(0, len(columns))]
    elif len(params) < len(columns):
        params = params + [[] for _ in range(len(params), len(columns))]

    for col, par in zip(columns, params):
        result, m, p, feat = onehot(result, col + move, features[col + move], par)
        computed.append(p)
        features = features[:col + move] + feat + features[col + move:]
        move += m
    return result, features, computed

def standardize(data: np.ndarray, column: int, mean: float = None, std: float = None) -> Tuple[np.ndarray, list]:
    result = np.copy(data)
    col = result[:, column]
    if mean is None:
        mean = col.mean()
    if std is None:
        std = col.std()
    result[:, column] = (col - mean) / std
    return result, [mean, std]

def normalize(data: np.ndarray, column: int, cmin: float = None, cmax: float = None) -> Tuple[np.ndarray, list]:
    result = np.copy(data)
    col = result[:, column]
    if cmin is None:
        cmin = col.min()
    if cmax is None:
        cmax = col.max()
    result[:, column] = (col - cmin) / (cmax - cmin)
    return result, [cmin, cmax]

def standardize_columns(data: np.ndarray, columns: list, method: str = 'std', params: list = []) -> Tuple[np.ndarray, list]:
    result = np.copy(data)
    computed = []

    if not params:
        params = [[None, None] for _ in range(0, len(columns))]
    elif len(params) < len(columns):
        params = params + [[None, None] for _ in range(len(params), len(columns))]

    for col, par in zip(columns, params):
        if method == 'std':
            result, p = standardize(result, col, par[0], par[1])
            computed.append(p)
        elif method == 'norm':
            result, p = normalize(result, col, par[0], par[1])
            computed.append(p)
    return result, computed

def feature_importance(weights: np.ndarray, k: int = 5) -> np.ndarray:
    if k > weights.size:
        k = weights.size
    result = np.hstack(
        [np.arange(0, weights.size - 1).reshape((-1, 1)),
        weights[1:]]
    )
    indices = np.abs(weights[1:]).flatten().argsort()[::-1]
    return result[indices][:k]
