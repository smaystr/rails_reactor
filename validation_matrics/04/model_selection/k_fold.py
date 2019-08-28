import math
import numpy as np
from typing import Union, List, Tuple


class KFold:
    """
    Usage:
    # kf = KFold()
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    """
    def __init__(
        self,
        n_splits: int = 3,
        shuffle: bool = True,
        random_state: Union[int, None] = None
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        if random_state is not None:
            np.random.seed(random_state)

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self,
        X: Union[np.ndarray, List, Tuple],
        y: Union[np.ndarray, List, Tuple, None] = None,
        groups: Union[np.ndarray, List, Tuple, None] = None,
    ):
        # TODO add groups param
        assert self.n_splits < len(X)
        if y is not None:
            assert len(X) == len(y)

        indexes = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(indexes)

        num_test_samples = math.trunc(len(X) / self.n_splits)
        start_test_index = 0
        finish_test_index = num_test_samples

        for n in range(self.n_splits):
            test_indexes = indexes[start_test_index:finish_test_index]
            train_indexes = np.setdiff1d(indexes, test_indexes)
            yield train_indexes, test_indexes

            start_test_index += num_test_samples
            finish_test_index += num_test_samples
