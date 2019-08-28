import numpy as np


def read_data(path, X_Y=True, dtype=np.float32):
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=dtype)
    if X_Y:
        return data[:, :-1], data[:, -1]
    return data


def ohe(data, column, params=None):
    if params is None:
        params = []
    a = data[:, column].astype(np.int32)

    if not params:
        un = np.unique(a)
    else:
        un = params

    b = np.zeros((a.size, un.shape[0]))
    b[np.arange(a.size), a - un.min()] = 1

    ret = np.zeros(shape=(data.shape[0], data.shape[1] + b.shape[1] - 1))
    ret[:, :column] = data[:, :column]
    ret[:, column:column + b.shape[1]] = b
    ret[:, column + b.shape[1]:] = data[:, column + 1:]
    return ret, un.shape[0] - 1, un


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


def ohe_columns(data, columns, params=[]):
    result = np.copy(data)
    computed = []
    move = 0

    if not params:
        params = [[] for _ in range(0, len(columns))]
    elif len(params) < len(columns):
        params = params + [[] for _ in range(len(params), len(columns))]

    for col, par in zip(columns, params):
        result, m, p = ohe(result, col + move, par)
        computed.append(p)
        move += m
    return result, computed


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
