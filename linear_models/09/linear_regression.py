import numpy as np
import math
from general_model import GeneralModel


class LinearRegression(GeneralModel):
    def __init__(self, batch_size=64, learning_rate=.005):
        self.coef = None
        self.x_train = None
        self.y_train = None
        self.loss = math.inf
        self.batch_size = batch_size
        self.epoch_num = 1
        self.n_train = None
        self.mean = None
        self.std = None
        self.learning_rate = learning_rate

    def _predict(self, X):
        return np.dot(X, self.coef)

    def fit(self, X, y, epochs=100):
        N, k_num = X.shape
        N_y,  = y.shape

        assert N == N_y

        self.n_train = N
        self.y_train = y
        normalized_X, self.std, self.mean = GeneralModel._normalize(X)
        self.x_train = GeneralModel._add_intercept(normalized_X)
        self.coef = GeneralModel._init_coef(k_num)
        self.batch_size = min(self.batch_size, N)

        while self.loss > 0.0001 and self.epoch_num < epochs:
            self._train()

        print(f'done! loss={self.loss}, converged in {self.epoch_num} epochs')

        return self

    def _train(self):
        for X_train_batch, y_train_batch in GeneralModel._get_minibatch(self.x_train, self.y_train, self.batch_size):
            y_pred = self._predict(X_train_batch)
            self.loss = self._get_loss(y_pred, y_train_batch)
            self.coef -= self.learning_rate * self._get_gradients(X_train_batch, y_train_batch, y_pred)

        self.epoch_num += 1

    def _get_loss(self, y, y_pred):
        return (1 / (2 * y.shape[0])) * np.sqrt(np.sum((y_pred - y)**2))

    def _get_gradients(self, X, y, y_pred):
        return (1 / X.shape[0]) * (np.dot(np.transpose(X), y_pred - y))

    def predict(self, X):
        N, k_num = X.shape
        assert k_num == self.coef.shape[0] - 1

        normalized_X = (X-self.mean) / self.std

        return self._predict(GeneralModel._add_intercept(normalized_X))


# X = np.array([[-1, 1], [1, 2], [2, 2], [2, 3], [4, 6], [5, 1], [3, 9], [5, 6]])
# y = 1 * x_0 + 2 * x_1 + 3
# y = np.dot(X, np.array([1, 2])) + 3
# clf = LinearRegression(batch_size=8).fit(X,y, 1000)
# print(clf.predict(np.array([[3, 5]])))
