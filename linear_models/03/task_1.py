import numpy as np
import argparse

import metrics
from preprocessing import MinMaxScalar


class LogRegression():

    def __init__(self, learning_rate, max_iter, verbose=False):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        self.X = X
        self.y = y

        self.theta = np.zeros(self.X.shape[1])

        if self.verbose:
            print("Loss:")

        for i in range(self.max_iter):
            z = self.X @ self.theta
            h = self.__sigmoid(z)
            gradient = self.X.T @ (h - self.y) / self.y.size

            self.theta -= self.lr * gradient

            if self.verbose and i % 100000 == 0:
                print(self.__loss(h, self.y))

        return self

    def predict(self, X):
        return self.__sigmoid(X @ self.theta) >= 0.5

    def predict_proba(self, X):
        return self.__sigmoid(X @ self.theta)

    def score(self, X_test, y_test, metric):
        return metric(self.predict(X_test), y_test)

    def get_theta(self):
        return self.theta


def main(lr, max_iter, std, metric):
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
        'heart_train.csv', delimiter=',', skip_header=True, dtype=int
    )
    X, y = data_train[:, :-1], data_train[:, -1]

    data_test = np.genfromtxt(
        'heart_test.csv', delimiter=',', skip_header=True, dtype=int
    )
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    if std:
        mms = MinMaxScalar()

        X = mms.fit_transform(X)
        X_test = mms.transform(X_test)

    clf = LogRegression(lr, max_iter)
    clf.fit(X, y)

    print(f"\n{metric} test:  {clf.score(X_test, y_test, metr[metric])}")
    print(f"{metric} train: {clf.score(X, y, metr[metric])}")
    print(f"\nFinal weights:\n {clf.get_theta()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heart Disease prediction"
        )
    parser.add_argument(
        '--learning_rate', help='Initialize precision', type=float,
        default=0.01, metavar='lr'
        )
    parser.add_argument(
        '--max_iter', help='Initialize number of iteration', type=int,
        default=100000
        )
    parser.add_argument(
        '--metric', help='Specify metrics. Available:\n accuracy, precision, \
recall, f1, mse, rmse, mae', type=str,
        default='accuracy'
        )

    def str2bool(v):
        return True if v.lower() in ('yes', 'true', 't', 'y', '1') else False

    parser.add_argument(
        '--std', help='Enable standartization', type=str2bool, default=True
        )

    args = parser.parse_args()
    main(args.learning_rate, args.max_iter, args.std, args.metric)
