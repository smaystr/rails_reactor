import numpy as np
import random

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help="dataset (path to local .csv file or url)")
    parser.add_argument('target', type=str, help="target variable name")
    parser.add_argument('task', type=str, choices=['classification', 'regression'], help="classification/regression")
    parser.add_argument('output', type=str, help="path for output model: output.info output.model")
    parser.add_argument('-s', '--split', type=str, default='train-test split',
                        choices=['train-test_split', 'k-fold', 'leave_one-out'],
                        help="parameter for validation split type")
    parser.add_argument('-vs', '--validation_size', type=float, default=None,
                        help="parameter for validation split size")
    parser.add_argument('-ts', '--time_series', type=str, default=None,
                        help="parameter for specifying time series column to perform timeseries validation")
    parser.add_argument('-hf', '--hyperparameter_fit', type=str, default='random search',
                        choices=['grid search', 'random search'],
                        help="parameter for hyperparameter fitting algo")
    args = parser.parse_args()

    return args


def get_data(args):
    '''
    :param args: arguments from parse_args
    :return: dataset, names of columns
    '''
    path_str = args.dataset
    data = np.genfromtxt(path_str, delimiter=',', skip_header=False, dtype=str)
    return data[1:], data[0]


def hold_out(dataset, target, test_size=0.2, random_state=228):
    random.seed(random_state)
    rows = random.sample(range(len(dataset)), int(test_size * len(dataset)))
    train_dataset = dataset[[row for row in range(len(dataset)) if not (row in rows)], :]
    test_dataset = dataset[rows, :]

    X_train = np.delete(train_dataset, target, 1)
    y_train = train_dataset[:, target]
    X_test = np.delete(test_dataset, target, 1)
    y_test = test_dataset[:, target]
    return X_train, y_train, X_test, y_test


def read_columns_names(path):
    with path.open(encoding='utf-8') as f:
        columns = f.readline()
    features = np.asarray(columns.strip().split(','))
    return features


def onehot_encoder(data, column, m=None):
    a = data[:, column].astype(dtype=np.int8)
    if m is None:
        m = np.unique(a).size
    b = np.eye(m)[a]

    ret = np.zeros((data.shape[0], data.shape[1] + m - 1))
    ret[:, :column] = data[:, :column]
    ret[:, column:column + m] = b
    ret[:, column + b.shape[1]:] = data[:, column + 1:]
    return ret, m


def onehoting_columns(data, columns, dimensions=None):
    result = np.copy(data)
    h = 0
    dim_ns = list()
    if dimensions is None:
        for column in columns:
            result, m = onehot_encoder(result, column + h)
            dim_ns.append(m)
            h += m - 1
    else:
        for col, d in zip(columns, dimensions):
            result, m = onehot_encoder(result, col + h, d)
            h += m - 1

    return result, dim_ns


def standartise(data, column, mean=None, std=None):
    res = np.copy(data)
    b = data[:, column]
    if mean is None:
        mean = b.mean()
    if std is None:
        std = b.std()
    res[:, column] = (b - mean) / std
    return res, (mean, std)


def standartise_columns(data, columns, mean_std=(None, None)):
    result = data[:]
    for column in columns:
        result, tup = standartise(result, column, mean_std[0], mean_std[1])

    return result, tup


def transform(data, columns, classes=None):
    result = data[:]
    if classes is None:
        classes = list()
        for column in columns:
            numeric_classes = np.unique(data[:, column])
            new_classes = {k: v for k, v in zip(numeric_classes, range(numeric_classes.size))}
            classes.append(new_classes)
            for key, value in new_classes.items():
                result[:, column][result[:, column] == key] = value
    else:
        for i in range(len(columns)):
            column = columns[i]
            new_classes = classes[i]
            for key, value in new_classes.items():
                result[:, column][result[:, column] == key] = value
    return result, classes


def prepare_data(train_data, columns_to_transform, columns_to_numeric=None):
    '''
    if we have str columns
    '''
    train = np.copy(train_data)
    if columns_to_numeric is not None:
        train, _ = transform(train, columns_to_numeric)
        train = train.astype(dtype=np.float32)

    return prepare(train, columns_to_transform)


def prepare(train_data, columns_to_transform):
    '''
    if all columns is int
    '''
    data_train, tup = standartise_columns(train_data, columns_to_transform['to_standartise'])
    train, dimensions = onehoting_columns(data_train, columns_to_transform['to_onehot_encoding'])
    print(dimensions)

    return train
