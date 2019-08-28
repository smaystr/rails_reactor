import numpy
from requests import request
from preprocessing import drop
from time import time


def sigmoid(z: numpy.ndarray):
    return 1 / (1 + numpy.power(numpy.e, -z))


def upload_dataset(path):
    """
    The method, which returns the data by the URL or by the local file
    :type path: str
    """
    if path.startswith('http://'):
        response = request('get', path)
        if response.status_code != 200:
            raise AttributeError(f"Status code: {response.status_code}. {response.headers}")
        else:
            return response.text
    elif path.endswith('.csv'):
        return numpy.genfromtxt(path, delimiter=',', skip_header=False, dtype=numpy.dtype)
    else:
        raise AttributeError(f"Error! {path} file format is unknown!")


def get_X_y(data, column):
    """
    Get X, y matrices, where column is a target
    :type data:numpy.ndarray
    :type column: str
    """
    if not is_string_type(data[0, :]):
        columns_titles = [column_title.decode('UTF-8') for column_title in data[0, :]]
    else:
        columns_titles = [column_title for column_title in data[0, :]]
    X = numpy.empty((data.shape[0] - 1, data.shape[1] - 1), dtype=object)
    y = numpy.empty((data.shape[0] - 1, 1), dtype=object)
    X_index = 0
    for index, column_title in enumerate(columns_titles):
        if column_title == column:
            y = numpy.asarray(data[1:len(data), index], dtype=numpy.float32)
        else:
            if is_numerical_type(data[1:len(data), index]):
                X[:, X_index] = numpy.asarray(data[1:len(data), index], dtype=numpy.float32)
                X_index += 1
            else:
                raise AttributeError(f'Error! Need one-hot encoding before getting X and y data.')
    return numpy.asarray(X, dtype=numpy.float32), y.reshape(-1, 1)


def benchmark_prediction():
    def _measure_time(func):
        def _wrapper(*args, **kwargs):
            start_time = time()
            loss, prediction = func(*args, **kwargs)
            return loss, prediction, time() - start_time
        return _wrapper
    return _measure_time


def is_string_type(column):
    return all(isinstance(x, str) for x in column)


def is_numerical_type(column):
    try:
        column.astype(numpy.float32)
    except Exception as exception:
        return False
    return True