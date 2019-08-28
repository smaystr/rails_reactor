from linear_regression import LinearRegression
from preprocessing import read_data, prepare_numeric_data
from metrics import rmse, mae
from argparse import ArgumentParser

import numpy as np
import pathlib

TRAIN = 'insurance_train.csv'
TEST = 'insurance_test.csv'

columns_to_transform = {'to_standartise': [0, 2, 3, -1],
                        'to_onehot_encoding': [5]}

columns_to_numeric = [1, 4, 5]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    path = pathlib.Path(parser.parse_args().path)

    train = read_data(path / TRAIN, type=str)
    test = read_data(path / TEST, type=str)

    train_x, test_x = prepare_numeric_data(train[:, :-1], test[:, :-1], columns_to_transform, columns_to_numeric)
    train_y = train[:, -1].reshape((-1, 1)).astype(dtype=np.float64)
    test_y = test[:, -1].reshape((-1, 1)).astype(dtype=np.float64)

    model = LinearRegression(lr=1e-5, epoch=10000).fit(train_x, train_y)
    print(f"rmse:{'%.2f' % model.score(test_x, test_y, rmse)}")
    print(f"accuracy:{'%.2f' % model.score(test_x, test_y, mae)}")
