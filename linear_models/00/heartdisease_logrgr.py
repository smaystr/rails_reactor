import logging
import pandas as pd
import numpy as np

from pathlib import Path

logger = logging.getLogger(__name__)

class LogisticRegression:

    def __init__(self, learning_rate = 0.01, iters = 1000):
        self._learning_rate = learning_rate
        self._iters = iters
        self.x_mean = None
        self.x_stddev = None
        self._theta = None
        self._bias = None

    def data_y(self, data):
        y = data.iloc[:, -1]
        return y[:, np.newaxis]

    def data_x(self, data):
        x = self._normalize(data[data.columns[:-1]]).iloc[:, 0:13]
        return np.c_[np.ones((x.shape[0], 1)), x]

    def initialize_weights(self, data_x):
        self._theta = np.random.rand(data_x.shape[1], 1)
        self._bias = np.zeros((1,))

    @staticmethod
    def _normalize(data):
        return (data - data.mean()) / data.std()

    @staticmethod
    def _sigmoid(z):
        # Activation function used to map any real value between 0 and 1
        return 1.0/(1.0 + np.exp(-1.0 * z))

    def _linear(self, data_x):
        return (data_x @ self._theta) + self._bias

    def probability(self, data_x):
        # Returns the probability after passing through sigmoid
        return self._sigmoid(self._linear(data_x))

    def fit(self, data):
        data_x = self.data_x(data)
        data_y = self.data_y(data)

        self.initialize_weights(data_x)

        self.x_mean = data_x.mean(axis=0).T
        self.x_stddev = data_x.std(axis=0).T

        # calculate_gradient
        for i in range(self._iters):
            # make normalized predictions
            diff = self.probability(data_x) - data_y

            # d/dw and d/db of mse
            delta_w = np.mean(diff * data_x, axis=0, keepdims=True).T
            delta_b = np.mean(diff)

            # update weights
            self._theta = self._theta - self._learning_rate * delta_w
            self._bias = self._bias - self._learning_rate * delta_b

        logger.info(f'Accuracy on train set: {self.accuracy(data_x, data_y) :.2f}')
        logger.info(f'Loss on train set: {self.loss(data_x, data_y) :.2f}')
        return self

    def predict(self, data_x):
        return (self.probability(data_x) >= 0.5).astype('int')

    def accuracy(self, data_x, data_y):
        predicts = self.predict(data_x)
        return np.mean(predicts == data_y)

    def loss(self, data_x, data_y):
        prblt = self.probability(data_x)
        pos_log = data_y * np.log(prblt + 1e-15)
        neg_log = (1 - data_y) * np.log((1 - prblt) + 1e-15)
        return -np.mean(pos_log + neg_log)


def main():
    data_path = Path('data')
    train_data = 'heart_train.csv'
    test_data = 'heart_test.csv'

    _format = '%(message)s'
    logging.basicConfig(level=logging.INFO, format=_format)

    df_train = pd.read_csv(data_path.joinpath(train_data))
    df_test = pd.read_csv(data_path.joinpath(test_data))

    model = LogisticRegression(1e-1, 1000)
    model.fit(df_train)

    x_test, y_test = model.data_x(df_test), model.data_y(df_test)
    logger.info(f'Accuracy on test set: {model.accuracy(x_test, y_test) :.2f}')
    logger.info(f'Loss on test set: {model.loss(x_test, y_test) :.2f}')


if __name__ == '__main__':
    main()
