import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from hw3.sklearn_benchmarking import test_sklearn_linreg
from hw3.utilities import *


class LinearRegression(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        learning_rate: float = 0.01,
        num_iterations: int = 100000,
        C=0.1,
        fit_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.num_iter = num_iterations
        self.C = C
        self.fit_intercept = fit_intercept
        self.weights = None

    @staticmethod
    def _add_intercept(x):
        return np.concatenate((np.ones((len(x), 1)), x), axis=1)

    def _loss(self, h, y):
        return (np.square(y - h)).mean() + self.C * (np.square(self.weights)).mean()

    def fit(self, x, y):
        logging.info('Fitting logreg')

        if self.fit_intercept:
            x = self._add_intercept(x)

        # weights initialization
        self.weights = np.random.normal(size=x.shape[1])[..., None]

        for i in range(self.num_iter):

            logit = np.dot(x, self.weights)

            loss = self._loss(logit, y)

            gradient = np.dot(x.T, (logit - y)) / len(x) + (1 / (self.C * len(x))) * self.weights

            self.weights -= self.learning_rate * gradient

            if i % 10000 == 0:
                logging.info(f'MSE: {loss}, RMSE {np.sqrt(loss)} \t')

        return self

    def predict(self, x):
        if self.fit_intercept:
            x = self._add_intercept(x)

        return np.dot(x, self.weights)

    def mse(self, h, y):

        if self.fit_intercept:
            h = self._add_intercept(h)

        logit = np.dot(h, self.weights)
        return self._loss(logit, y)


def test_lin_reg(data: Dataset):
    lin_reg = LinearRegression(
        learning_rate=0.01,
        num_iterations=100000,
        C=0.1
    )

    lin_reg.fit(data.X_train, data.Y_train)
    logging.info(f'My model score for test data is: {lin_reg.score(data.X_test, data.Y_test)}')
    logging.info(f'My model score for train data is: {lin_reg.score(data.X_train, data.Y_train)}')
    logging.info(f'My model MSE for test data is : {lin_reg.mse(data.X_test, data.Y_test)}')
    logging.info(f'My model MSE for train data is : {lin_reg.mse(data.X_train, data.Y_train)}')


if __name__ == '__main__':
    args = parse_args()

    if args['verbose']:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='info.log',
                            filemode='w')
    dataset = Dataset(
        force_download=args['force_download'],
        dataset_path=args['dataset_path'],
        test_url=args['test_url'],
        train_url=args['train_url'],
        categorical_features=['sex', 'smoker', 'region'],
        target_column='charges'
    )
    try:
        dataset.load_dataset()
    except DatasetException:
        logging.error('DATASET ERROR! SHUTTING DOWN APP')
        exit(1)

    test_lin_reg(dataset)
    test_sklearn_linreg(dataset)
