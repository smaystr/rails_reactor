import logging
import numpy as np

logger = logging.getLogger(__name__)


class LinearRegression:
    def __init__(self, learning_rate=0.01, iters=1000):
        self._learning_rate = learning_rate
        self._iters = iters
        self._theta = None
        self._bias = None

    def initialize_weights(self, data_x):
        self._theta = np.random.rand(data_x.shape[1], 1)
        self._bias = np.zeros((1,))

    def predicts(self, data_x):
        return (data_x @ self._theta) + self._bias

    def fit(self, x_train, y_train):
        self.initialize_weights(x_train)

        # calculate_gradient
        for i in range(self._iters):
            # make normalized predictions
            # find error
            dff = self.predicts(x_train) - y_train

            # compute d/dw and d/db of MSE
            delta_w = np.mean(dff * x_train, axis=0, keepdims=True).T
            delta_b = np.mean(dff)

            # update weights and biases
            self._theta = self._theta - self._learning_rate * delta_w
            self._bias = self._bias - self._learning_rate * delta_b
        return self

    def predict(self, data_x, data_y):
        return self.predicts(data_x) * data_y.std() + data_y.mean()

    @staticmethod
    def calculate_mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
