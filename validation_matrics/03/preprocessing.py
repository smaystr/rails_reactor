import math
import numpy
import utils


def fillna(X, columns=None, method='mean'):
    """
    If the NaN values are in the column, fill them with average value
    :type X: numpy.ndarray
    :type columns: list
    :param method: str
    """
    copied_data = numpy.copy(X)
    if columns is None:
        columns = list(range(len(copied_data.T)))
    has_nans = []
    for column, _ in enumerate(copied_data.T):
        if column in columns:
            copied_data, has_nan = fillna_column(copied_data, column, method=method)
            has_nans.append(has_nan)
    return copied_data, has_nans


def fillna_column(X, column, method='mean'):
    """
    If the NaN values are in the column, fill them with average value
    :type X: numpy.ndarray
    :type column: int
    :param method: str
    """
    copied_data = numpy.copy(X)
    column_data = copied_data[:, column]
    not_nan_values = [value for value in column_data if not math.isnan(value)]
    if method == 'mean':
        value = sum(not_nan_values) / len(not_nan_values)
    elif method == 'zeros':
        value = 0
    elif method == 'min':
        value = numpy.min(not_nan_values)
    elif method == 'max':
        value = numpy.max(not_nan_values)
    has_nan = False
    for row, value in enumerate(column_data):
        if math.isnan(value):
            has_nan = True
            column_data[row] = value
    copied_data[:, column] = column_data
    return copied_data, has_nan


def dropna(X, columns=None):
    """
    If the NaN values are in the X matrix's column, drop it from the copy of the X matrix
    :type X: numpy.ndarray
    :type columns: list
    """
    copied_data = numpy.copy(X)
    if columns is None:
        columns = list(range(len(copied_data.T)))
    has_nans = []
    for column in columns:
        copied_data, has_nan = dropna_column(copied_data, column)
        if has_nan:
            columns.pop()
        has_nans.append(has_nan)
    return copied_data, has_nans


def dropna_column(X, column):
    """
    If the NaN values are in the X matrix's column, drop it from the copy of the X matrix
    :type X: numpy.ndarray
    :type column: int
    """
    copied_data = numpy.copy(X)
    column_data = copied_data[:, column]
    has_nan = False
    for value in column_data:
        if math.isnan(value):
            has_nan = True
            break
    if has_nan:
        copied_data = numpy.delete(copied_data, column, axis=1)
    return copied_data, has_nan


def drop(X, columns=None):
    """
    Drop the column from the copy of X matrix
    :type X: numpy.ndarray
    :type columns: list
    """
    copied_data = numpy.copy(X)
    columns = numpy.array(columns)
    if columns is None:
        return copied_data
    else:
        for column in columns:
            copied_data = numpy.delete(copied_data, column, axis=1)
            columns -= 1
        return copied_data


def normalize(data, columns=None, mins=None, maxs=None):
    """
    Feature normalization (aka Min-Max scaling)
    :type data: numpy.ndarray
    :type columns: list
    :param maxs: list
    :param mins: list
    """
    copied_data = numpy.copy(data)
    if columns is None:
        columns = list(range(len(copied_data.T)))
    columns_mins = [None for iteration in range(len(columns))] if mins is None else mins
    columns_maxs = [None for iteration in range(len(columns))] if maxs is None else maxs
    for iteration, column in enumerate(columns):
        copied_data, *_ = normalize_column(copied_data, column, columns_mins[iteration], columns_maxs[iteration])
    return copied_data


def normalize_column(data, column, minimum=None, maximum=None):
    """
    Feature normalization (aka Min-Max scaling)
    :type data: numpy.ndarray
    :type column: int
    :param maximum: float
    :param minimum: float
    """
    copied_data = numpy.copy(data)
    column_data = copied_data[:, column]
    column_minimum = column_data.min() if minimum is None else minimum
    column_maximum = column_data.max() if maximum is None else maximum
    copied_data[:, column] = (column_data - column_minimum) / (column_maximum - column_minimum)
    return copied_data, column_minimum, column_maximum


def standardize(data, columns=None, means=None, stds=None):
    """
    Feature normalization (aka Min-Max scaling)
    :type data: numpy.ndarray
    :param columns: list
    :param means: list
    :param stds: list
    """
    copied_data = numpy.copy(data)
    if columns is None:
        columns = list(range(len(copied_data.T)))
    columns_means = [None for iteration in range(len(columns))] if means is None else means
    columns_stds = [None for iteration in range(len(columns))] if stds is None else stds
    for column, _ in enumerate(columns):
        copied_data, *_ = standardize_column(copied_data, column, columns_means[column], columns_stds[column])
    return copied_data


def standardize_column(data, column, mean=None, std=None):
    """
    Feature standardization
    :type data: numpy.ndarray
    :type column: int
    :param mean: float
    :param std: float
    """
    copied_data = numpy.copy(data)
    column_data = copied_data[:, column]
    column_mean = column_data.mean() if mean is None else mean
    column_std = column_data.std() if std is None else std
    copied_data[:, column] = (column_data - column_mean) / column_std
    return copied_data, column_mean, column_std


def one_hot_encoding(data, columns):
    """
    One-hot encoding
    :type data: numpy.ndarray
    :param columns: list
    """
    column_titles = [column_title.decode('UTF-8') for column_title in data[0, :]]
    columns_indices = [column_index for column_index, _ in enumerate(data[0, :]) if data[0, column_index].decode('UTF-8') in columns]
    copied_data = numpy.copy(data[1:len(data), :])
    columns_to_drop = []
    for column_title, column_index in zip(columns, columns_indices):
        if utils.is_numerical_type(column_title):
            copied_data[:, column_index] = column_title.astype(numpy.float32)
        else:
            unique_values, unique_indices = numpy.unique(copied_data[:,column_index], return_inverse=True)
            if len(unique_values) / len(data) > 0.2:
                raise AttributeError(
                    f'Your column by index {column_index} contains too much unique elements! Drop it or delete it manually.')
            column_titles += [values.decode('UTF-8') for values in unique_values]
            category_matrix = numpy.zeros((len(copied_data), len(unique_values)))
            category_matrix[numpy.arange(len(copied_data)), unique_indices] = 1
            copied_data = numpy.concatenate((copied_data, category_matrix), axis=1)
            columns_to_drop.append(column_index)
    copied_data = drop(copied_data, columns=columns_to_drop)
    copied_data = numpy.asarray(copied_data, dtype=numpy.float32)
    for title in columns:
        column_titles.remove(title)
    column_titles = numpy.asarray(column_titles, dtype=object).reshape(-1, 1).T
    copied_data = numpy.concatenate((column_titles, copied_data), axis=0)
    return copied_data

import math
import numpy
import utils


def fillna(X, columns=None, method='mean'):
    """
    If the NaN values are in the column, fill them with average value
    :type X: numpy.ndarray
    :type columns: list
    :param method: str
    """
    copied_data = numpy.copy(X)
    if columns is None:
        columns = list(range(len(copied_data.T)))
    has_nans = []
    for column, _ in enumerate(copied_data.T):
        if column in columns:
            copied_data, has_nan = fillna_column(copied_data, column, method=method)
            has_nans.append(has_nan)
    return copied_data, has_nans


def fillna_column(X, column, method='mean'):
    """
    If the NaN values are in the column, fill them with average value
    :type X: numpy.ndarray
    :type column: int
    :param method: str
    """
    copied_data = numpy.copy(X)
    column_data = copied_data[:, column]
    not_nan_values = [value for value in column_data if not math.isnan(value)]
    if method == 'mean':
        value = sum(not_nan_values) / len(not_nan_values)
    elif method == 'zeros':
        value = 0
    elif method == 'min':
        value = numpy.min(not_nan_values)
    elif method == 'max':
        value = numpy.max(not_nan_values)
    has_nan = False
    for row, value in enumerate(column_data):
        if math.isnan(value):
            has_nan = True
            column_data[row] = value
    copied_data[:, column] = column_data
    return copied_data, has_nan


def dropna(X, columns=None):
    """
    If the NaN values are in the X matrix's column, drop it from the copy of the X matrix
    :type X: numpy.ndarray
    :type columns: list
    """
    copied_data = numpy.copy(X)
    has_nans = []
    if columns is None:
        columns = list(range(len(copied_data.T)))
    if columns is int:
        column_data = copied_data[:, columns]
        has_nan = False
        for value in column_data:
            if math.isnan(value):
                has_nan = True
                break
        if has_nan:
            copied_data = numpy.delete(copied_data, columns, axis=1)
        has_nans.append(has_nan)
    if columns is list:
        for column in columns:
            column_data = copied_data[:, column]
            has_nan = False
            for value in column_data:
                if math.isnan(value):
                    has_nan = True
                    break
            if has_nan:
                copied_data = numpy.delete(copied_data, columns, axis=1)
                columns.pop()
            has_nans.append(has_nan)
    return copied_data, has_nans


def drop(X, columns=None):
    """
    Drop the column from the copy of X matrix
    :type X: numpy.ndarray
    :type columns: list
    """
    copied_data = numpy.copy(X)
    columns = numpy.array(columns)
    if columns is None:
        return copied_data
    else:
        for column in columns:
            copied_data = numpy.delete(copied_data, column, axis=1)
            columns -= 1
        return copied_data


def normalize(data, columns=None, mins=None, maxs=None):
    """
    Feature normalization (aka Min-Max scaling)
    :type data: numpy.ndarray
    :type columns: list
    :param maxs: list
    :param mins: list
    """
    copied_data = numpy.copy(data)
    if columns is None:
        columns = list(range(len(copied_data.T)))
    columns_mins = [None for iteration in range(len(columns))] if mins is None else mins
    columns_maxs = [None for iteration in range(len(columns))] if maxs is None else maxs
    for iteration, column in enumerate(columns):
        copied_data, *_ = normalize_column(copied_data, column, columns_mins[iteration], columns_maxs[iteration])
    return copied_data


def normalize_column(data, column, minimum=None, maximum=None):
    """
    Feature normalization (aka Min-Max scaling)
    :type data: numpy.ndarray
    :type column: int
    :param maximum: float
    :param minimum: float
    """
    copied_data = numpy.copy(data)
    column_data = copied_data[:, column]
    column_minimum = column_data.min() if minimum is None else minimum
    column_maximum = column_data.max() if maximum is None else maximum
    copied_data[:, column] = (column_data - column_minimum) / (column_maximum - column_minimum)
    return copied_data, column_minimum, column_maximum


def standardize(data, columns=None, means=None, stds=None):
    """
    Feature normalization (aka Min-Max scaling)
    :type data: numpy.ndarray
    :param columns: list
    :param means: list
    :param stds: list
    """
    copied_data = numpy.copy(data)
    if columns is None:
        columns = list(range(len(copied_data.T)))
    columns_means = [None for iteration in range(len(columns))] if means is None else means
    columns_stds = [None for iteration in range(len(columns))] if stds is None else stds
    for column, _ in enumerate(columns):
        copied_data, *_ = standardize_column(copied_data, column, columns_means[column], columns_stds[column])
    return copied_data


def standardize_column(data, column, mean=None, std=None):
    """
    Feature standardization
    :type data: numpy.ndarray
    :type column: int
    :param mean: float
    :param std: float
    """
    copied_data = numpy.copy(data)
    column_data = copied_data[:, column]
    column_mean = column_data.mean() if mean is None else mean
    column_std = column_data.std() if std is None else std
    copied_data[:, column] = (column_data - column_mean) / column_std
    return copied_data, column_mean, column_std


def one_hot_encoding(data, columns):
    """
    One-hot encoding
    :type data: numpy.ndarray
    :param columns: list
    """
    column_titles = [column_title.decode('UTF-8') for column_title in data[0, :]]
    columns_indices = [column_index for column_index, _ in enumerate(data[0, :]) if data[0, column_index].decode('UTF-8') in columns]
    copied_data = numpy.copy(data[1:len(data), :])
    columns_to_drop = []
    for column_title, column_index in zip(columns, columns_indices):
        if utils.is_numerical_type(column_title):
            copied_data[:, column_index] = column_title.astype(numpy.float32)
        else:
            unique_values, unique_indices = numpy.unique(copied_data[:,column_index], return_inverse=True)
            if len(unique_values) / len(data) > 0.2:
                raise AttributeError(
                    f'Your column by index {column_index} contains too much unique elements! Drop it or delete it manually.')
            column_titles += [values.decode('UTF-8') for values in unique_values]
            category_matrix = numpy.zeros((len(copied_data), len(unique_values)))
            category_matrix[numpy.arange(len(copied_data)), unique_indices] = 1
            copied_data = numpy.concatenate((copied_data, category_matrix), axis=1)
            columns_to_drop.append(column_index)
    copied_data = drop(copied_data, columns=columns_to_drop)
    copied_data = numpy.asarray(copied_data, dtype=numpy.float32)
    for title in columns:
        column_titles.remove(title)
    column_titles = numpy.asarray(column_titles, dtype=object).reshape(-1, 1).T
    copied_data = numpy.concatenate((column_titles, copied_data), axis=0)
    return copied_data

