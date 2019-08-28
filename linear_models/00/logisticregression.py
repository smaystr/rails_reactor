import logging
import numpy as np

from linearregression import LinearRegression

logger = logging.getLogger(__name__)


class LogisticRegression(LinearRegression):
    def __init__(self, learning_rate=0.01, iters=1000):
        super().__init__(learning_rate, iters)

    @staticmethod
    def _sigmoid(z):
        # Activation function used to map any real value between 0 and 1
        return 1.0 / (1.0 + np.exp(-1.0 * z))

    def predicts(self, data_x):
        # Returns the probability after passing through sigmoid
        return self._sigmoid((data_x @ self._theta) + self._bias)

    def predict(self, data_x, data_y):
        return (self.predicts(data_x) >= 0.5).astype('int')

    def accuracy(self, data_x, data_y):
        _predicts = self.predict(data_x, data_y)
        return np.mean(_predicts == data_y)

    def loss(self, data_x, data_y):
        probability = self.predicts(data_x)
        pos_log = data_y * np.log(probability + 1e-15)
        neg_log = (1 - data_y) * np.log((1 - probability) + 1e-15)
        return -np.mean(pos_log + neg_log)
