import numpy as np
import argparse

import metrics
from preprocessing import MinMaxScalar


class LinearRegressor():

    def __init__(self, learning_rate, max_iter, verbose=False):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose

    def __loss(self, y_hat, y):
        return np.sqrt(np.mean(np.square(y_hat - y)))

    def fit(self, X, y):
        self.theta = np.zeros((X.shape[1] + 1, 1), dtype=np.float32)

        self.y = y.reshape((-1, 1))
        self.X = np.ones((X.shape[0], X.shape[1] + 1))
        self.X[:, 1:] = X

        if self.verbose:
            print("Loss:")

        for i in range(self.max_iter):
            y_hat = self.predict(X)
            grad = self.X.T @ (y_hat - self.y)
            self.theta -= self.lr * grad / X.shape[0]

            if self.verbose and i % 10000 == 0:
                print(self.__loss(y_hat, self.y))

        return self

    def predict(self, X):
        X_ = np.ones((X.shape[0], X.shape[1] + 1))
        X_[:, 1:] = X
        return X_.dot(self.theta)

    def score(self, X, y, metric):
        return metric(self.predict(X), y.reshape((-1, 1)))

    def get_theta(self):
        return self.theta[1:]  # not including bias


def feature_transform(X):
    classes = {'northwest': 1, 'northeast': 2, 'southwest': 3, 'southeast': 4}

    for row in X:
        row[1] = 0 if row[1] == 'male' else 1
        row[4] = 0 if row[4] == 'no' else 1
        row[5] = classes[row[5]]

    return X.astype(np.float32)


def main(lr, max_iter, metric, std):
    metr = {
        'accuracy': metrics.accuracy,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'mse': metrics.mse,
        'rmse': metrics.rmse,
        'mae': metrics.mae,
        'f1': metrics.f1
    }

    data_train = np.genfromtxt(
        'insurance_train.csv', delimiter=',', skip_header=1, dtype=str
    )
    X, y = data_train[:, :-1], data_train[:, -1]

    X = feature_transform(X)
    y = y.astype(np.float32)

    data_test = np.genfromtxt(
        'insurance_test.csv', delimiter=',', skip_header=1, dtype=str
    )
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    X_test = feature_transform(X_test)
    y_test = y_test.astype(np.float32)

    if std:
        mms = MinMaxScalar()

        X = mms.fit_transform(X)
        X_test = mms.transform(X_test)

    reg = LinearRegressor(lr, max_iter)
    reg.fit(X, y)

    print(f"\n{metric} test:  {reg.score(X_test, y_test, metr[metric])}")
    print(f"{metric} train: {reg.score(X, y, metr[metric])}")
    print(f"\nFinal weights:\n {reg.get_theta().astype(int)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Cost prediction"
        )
    parser.add_argument(
        '--learning_rate', help='Initialize precision', type=np.float32,
        default=0.01, metavar='lr'
        )
    parser.add_argument(
        '--max_iter', help='Initialize number of iteration', type=np.int32,
        default=100000
        )
    parser.add_argument(
        '--metric', help='Specify metrics. Available:\n accuracy, precision, \
recall, f1, mse, rmse, mae', type=str,
        default='rmse'
        )

    def str2bool(v):
        return True if v.lower() in ('yes', 'true', 't', 'y', '1') else False

    parser.add_argument(
        '--std', help='Enable standartization', type=str2bool, default=True
        )

    args = parser.parse_args()
    main(args.learning_rate, args.max_iter, args.metric, args.std)
