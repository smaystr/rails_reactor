from logistic_regression import LogisticRegression
from preprocessing import read_data, prepare_data
from metrics import accuracy, F1
from argparse import ArgumentParser

import pathlib

TRAIN = 'heart_train.csv'
TEST = 'heart_test.csv'


columns = {'to_standartise': [0, 3, 4, 7, 9, ],
           'to_onehot_encoding': [2, 6, 11, 12]}

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    path = pathlib.Path(parser.parse_args().path)

    train = read_data(path / TRAIN)
    test = read_data(path / TEST)

    train_x, test_x, = prepare_data(train[:, :-1], test[:, :-1], columns)
    train_y = train[:, -1].reshape((-1, 1))
    test_y = test[:, -1].reshape((-1, 1))

    model = LogisticRegression().fit(train_x, train_y)
    print(f"f1 score:{'%.2f' % model.score(test_x, test_y, F1)}")
    print(f"accuracy:{'%.2f'%model.score(test_x, test_y, accuracy)}")
