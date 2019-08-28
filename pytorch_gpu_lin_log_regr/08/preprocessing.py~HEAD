import json
import sys
import numpy as np


def read_data(path, X_Y=True, dtype=np.float32):
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=dtype)
    if X_Y:
        return data[:, :-1], data[:, -1]
    return data


def read_model_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    need_keys = {'lr', 'batch', 'penalty', 'C', 'epoch', 'cuda'}
    if not need_keys <= data.keys():
        print(f"Check your config file. You need to add {set(need_keys).difference(set(data.keys()))} field(s)")
        sys.exit()
    return data


def read_feature_names(path, skip_last=True) -> np.ndarray:
    with path.open(encoding='utf-8') as f:
        features = f.readline()
    features = np.asarray(features.strip().split(','))
    if skip_last:
        return features[:-1]
    return features


def ohe(data, column, feature, params):
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
    result[:, column + hotted.shape[1]:] = data[:, column + 1:]
    feat_list = [f'{feature}_{un}' for un in unique]
    return result, unique.shape[0] - 1, unique, feat_list


def to_numeric(data, column, classes):
    result = np.copy(data)
    for k, v in classes.items():
        result[:, column][result[:, column] == k] = v
    return result


def to_numeric_multiple(data, columns):
    classes_list = []
    result = np.copy(data)
    for c in columns:
        un = np.unique(data[:, c])
        classes = {k: v for k, v in zip(un, range(un.shape[0]))}
        classes_list.append(classes)
        for k, v in classes.items():
            result[:, c][result[:, c] == k] = v
    return result, classes_list


def ohe_columns(data, columns, features, params=None):
    result = np.copy(data)
    computed = []
    move = 0

    if not params:
        params = [[] for _ in range(0, len(columns))]
    elif len(params) < len(columns):
        params = params + [[] for _ in range(len(params), len(columns))]

    for col, par in zip(columns, params):
        result, m, p, feat = ohe(
            result, col + move, features[col + move], par)
        computed.append(p)
        features = np.concatenate([features[:col + move], feat, features[col + move:]])
        move += m
    return result, features, computed


def standardize(data, column, mean=None, std=None):
    result = np.copy(data)
    col = result[:, column]
    if mean is None:
        mean = col.mean()
    if std is None:
        std = col.std()
    result[:, column] = (col - mean) / std
    return result, [mean, std]


def normalize(data, column, cmin=None, cmax=None):
    result = np.copy(data)
    col = result[:, column]
    if cmin is None:
        cmin = col.min()
    if cmax is None:
        cmax = col.max()
    result[:, column] = (col - cmin) / (cmax - cmin)
    return result, [cmin, cmax]


def standardize_columns(data, columns, method='std', params=None):
    if params is None:
        params = []
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


def get_important_feats(weights, k=5):
    if k > weights.size:
        k = weights.size
    result = np.hstack(
        [np.arange(0, weights.size).reshape((-1, 1)),
         np.abs(weights).reshape((-1, 1))]
    )
    return result[result[:, 1].argsort()[::-1]][:k]
