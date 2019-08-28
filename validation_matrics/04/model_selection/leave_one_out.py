import numpy as np
from typing import Union, List, Tuple


class LeaveOneOut:
    """
    Usage:
    # lv = LeaveOneOut()
    # for train_index, test_index in lv.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    """
    def __init__(self) -> None:
        pass

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self,
        X: Union[np.ndarray, List, Tuple],
        y: Union[np.ndarray, List, Tuple, None] = None,
        groups: Union[np.ndarray, List, Tuple, None] = None,
    ):
        # TODO add groups param
        if y is not None:
            assert len(X) == len(y)

        indexes = np.arange(len(X))

        num_test_samples = 1
        start_test_index = 0
        finish_test_index = num_test_samples

        for n in range(len(X)):
            test_indexes = indexes[start_test_index:finish_test_index]
            train_indexes = np.setdiff1d(indexes, test_indexes)
            yield train_indexes, test_indexes

            start_test_index += num_test_samples
            finish_test_index += num_test_samples
