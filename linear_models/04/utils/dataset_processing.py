import numpy as np
from typing import Union, List, Tuple


def add_ones_column(data: np.ndarray) -> np.ndarray:
    data_processed = np.copy(data)
    num_examples = data_processed.shape[0]
    data_processed = np.c_[np.ones((num_examples, 1)), data_processed]
    return data_processed


def normalize_data(
    data: np.ndarray, columns: Union[None, List[int], Tuple[int]] = None
) -> np.ndarray:
    """
    Normalize features.
    Normalizes input features X. Returns a normalized version of X where
    the mean value of each feature is 0 and deviation is close to 1.
    :param data: set of features.
    :param columns: set of columns to normalize
    :return: normalized set of features.
    """
    data = np.copy(data).astype(float)
    if columns:
        initial_data = np.delete(data, columns, 1)
        data = data[:, columns]
    features_mean = np.mean(data, 0)
    features_deviation = np.std(data, 0)
    if data.shape[0] > 1:
        data -= features_mean
    # Normalize each feature values so that all features are close to [-1:1].
    # Also prevent division by zero error.
    features_deviation[features_deviation == 0] = 1
    data /= features_deviation
    if columns:
        data = np.hstack((initial_data, data))

    return data
