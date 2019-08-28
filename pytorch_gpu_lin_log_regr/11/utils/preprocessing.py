import numpy as np
import pandas as pd
import torch


def load_data(path_train, path_test, task, device):
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    k = df_train.values.shape[0]

    if task == 'Classification':
        columns_to_transform = {'to_standartise': [0, 3, 4, 7, 9],
                                'to_onehot_encoding': [2, 6, 10, 11, 12]}
    else:
        columns_to_transform = {'to_standartise': [0, 2, 3],
                                'to_onehot_encoding': [1, 4, 5]}

    train_x, train_y, test_x, test_y = prepare_data(df_train.values, df_test.values, columns_to_transform['to_standartise'])

    data = torch.tensor(
        pd.get_dummies(pd.DataFrame(np.vstack([train_x, test_x])),
                       columns=columns_to_transform['to_onehot_encoding']).values.astype(np.float64),
        dtype=torch.float32, device=device)

    return data[:k, :], torch.tensor(train_y, dtype=torch.float32, device=device), data[k:, :], torch.tensor(test_y,
                                                                                                             dtype=torch.float32,
                                                                                                             device=device)


def standartise(data, column, mean=None, std=None):
    res = np.copy(data)
    b = data[:, column]
    if mean is None:
        mean = b.mean()
    if std is None:
        std = b.std()
    res[:, column] = (b - mean) / std
    return res, mean, std


def standartise_columns(data, columns, columns_info=None):
    result = data[:]

    if columns_info is None:
        columns_info = []
        for column in columns:
            result, mean, std = standartise(result, column)
            columns_info.append((mean, std))
    else:
        for column, mean_std in zip(columns, columns_info):
            result, _, _ = standartise(result, column, mean_std[0], mean_std[1])

    return result, columns_info


def prepare_data(data_train, data_test, columns_to_standartise):
    train, columns_info = standartise_columns(data_train, columns_to_standartise)
    test, _ = standartise_columns(data_test, columns_to_standartise, columns_info=columns_info)


    return train[:, :-1], train[:, -1].reshape((-1, 1)).astype(np.float64), test[:, :-1], \
                                       test[:, -1].reshape((-1, 1)).astype(np.float64)
