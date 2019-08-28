import numpy as np
import pandas as pd
from typing import Union, List, Tuple

data_type = Union[np.ndarray, pd.DataFrame]


def train_test_split(
    X: data_type,
    y: data_type = None,
    train_size: int = 80,
    random_state: Union[None, int] = None,
    shuffle: bool = True,
    stratify: Union[None, List[int], Tuple[int], np.ndarray] = None,
) -> Tuple[data_type, data_type, data_type, data_type]:
    """
    :param shuffle: If shuffle=False then stratify must be None.
    """
    if y is not None:
        assert len(X) == len(y)

    if random_state is not None:
        np.random.seed(random_state)

    if shuffle and stratify is not None:
        assert len(X) == len(stratify)
        values, counts = np.unique(stratify, return_counts=True)
        train_counts = []
        if len(values) > 1:
            for count_index in range(len(counts) - 1):
                train_cnt = int(counts[count_index] * train_size / 100)
                train_counts.append(train_cnt)
            # to make sure, that we have all observation
            train_counts.append(int(len(stratify) * train_size / 100) - sum(train_counts))

            train_mask = []
            test_mask = []
            for val_index, uniq_val in enumerate(values):
                mask = np.where(stratify == uniq_val)[0]
                train_mask.append(mask[:train_counts[val_index]])
                test_mask.append(mask[train_counts[val_index]:])

            train_mask = np.concatenate(train_mask)
            test_mask = np.concatenate(test_mask)

            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_mask, :], X.iloc[test_mask, :]
            else:
                X_train, X_test = X[train_mask, :], X[test_mask, :]
            splitted_data = [X_train, X_test]
            if y is not None:
                if isinstance(y, pd.DataFrame):
                    y_train, y_test = y.iloc[train_mask], y.iloc[test_mask]
                else:
                    y_train, y_test = y[train_mask], y[test_mask]
                splitted_data += [y_train, y_test]
            return splitted_data
        else:
            pass
    elif shuffle:
        shuffle_indexes = np.random.permutation(X.shape[0])
        if isinstance(X, pd.DataFrame):
            X = X.iloc[shuffle_indexes]
        else:
            X = X[shuffle_indexes]
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[shuffle_indexes]
            else:
                y = y[shuffle_indexes]
    elif not shuffle and stratify is not None:
        raise Exception('train_test_split exception:\nIf shuffle=False then stratify must be None.')

    train_n = int(len(X) * train_size / 100)
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[:train_n, :], X.iloc[train_n:, :]
    else:
        X_train, X_test = X[:train_n, :], X[train_n:, :]
    splitted_data = [X_train, X_test]
    if y is not None:
        if isinstance(y, pd.DataFrame):
            y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]
        else:
            y_train, y_test = y[:train_n], y[train_n:]
        splitted_data += [y_train, y_test]
    return splitted_data
