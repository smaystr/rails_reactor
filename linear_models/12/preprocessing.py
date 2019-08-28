import numpy as np
import pathlib
from argparse import ArgumentParser


def read_data(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=np.float32)
    return data


def get_path():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    return pathlib.Path(parser.parse_args().path)


def onehot_encoder(data, column, m=None):
    '''
    data: data
    column: column number
    m: number of different classes in feature
    '''
    encodeing_column = data[:, column].astype(dtype=np.int8)
    if m is None:
        m = np.unique(encodeing_column).size
    new_encoded_columns = np.eye(m)[encodeing_column]

    ret = np.zeros((data.shape[0], data.shape[1] + m - 1))
    ret[:, :column] = data[:, :column]
    ret[:, column:column + m] = new_encoded_columns
    ret[:, column + new_encoded_columns.shape[1]:] = data[:, column+1:]
    return ret, m


def onehoting_columns(data, columns, dimensions=None):
    result = np.copy(data)
    step = 0
    dim_ns = list()
    if dimensions is None:
        for column in columns:
            result, m = onehot_encoder(result, column + step)
            dim_ns.append(m)
            step += m - 1
    else:
        for col, d in zip(columns, dimensions):
            result, m = onehot_encoder(result, col + step, d)
            step += m - 1

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


def standartise_columns(data, columns, mean_std=(None,None)):
    result = data[:]
    for column in columns:
        result, tup = standartise(result, column, mean_std[0], mean_std[1])

    return result, tup


def prepare_data(TRAIN, TEST, columns_to_transfrom):

    data_train, tup = standartise_columns(TRAIN, columns_to_transfrom['to_standartise'])
    train, dimensions = onehoting_columns(data_train, columns_to_transfrom['to_onehot_encoding'])

    data_test, _ = standartise_columns(TEST, columns_to_transfromr['to_standartise'], mean_std=tup)
    test, _ = onehoting_columns(data_test, columns_to_transfrom['to_onehot_encoding'], dimensions=dimensions)

    return train, test


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


def prepare_numeric_data(train_data, test_data, columns_to_transfrom, columns_to_numeric):

    numeric_train, classes = transform(train_data, columns_to_numeric)
    numeric_test, _ = transform(test_data, columns_to_numeric, classes=classes)

    numeric_train = numeric_train.astype(dtype=np.float32)
    numeric_test = numeric_test.astype(dtype=np.float32)

    train, tup = standartise_columns(numeric_train, columns_to_transfrom['to_standartise'])
    new_train, dimensions = onehoting_columns(train, columns_to_transfrom['to_onehot_encoding'])

    test, _ = standartise_columns(numeric_test, columns_to_transfrom['to_standartise'], mean_std=tup)
    new_test, _ = onehoting_columns(test, columns_to_transfrom['to_onehot_encoding'], dimensions=dimensions)

    return new_train, new_test
