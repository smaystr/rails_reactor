import numpy as np
import math
from general_model import GeneralModel

class LogisticRegression(GeneralModel):
    def __init__(self, batch_size=64, learning_rate=.0005):
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

        print('done', self.loss, self.epoch_num)
        print(self.coef)

        return self

    def _train(self):
        for X_train_batch, y_train_batch in GeneralModel._get_minibatch(self.x_train, self.y_train, self.batch_size):
            y_pred = self._proba(X_train_batch)
            self.loss = self._get_loss(y_train_batch, y_pred)
            self.coef -= self.learning_rate * self._get_gradients(X_train_batch, y_train_batch, y_pred)

        self.epoch_num += 1

    def _get_loss(self, y, y_pred):
        # https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11
        # cost = -y * np.log(y_pred) - (1-y) * np.log(1 - y_pred)
        m = y.shape[0]

        return (1 / m) * np.sum(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

    def _get_gradients(self, X, y, y_pred):
        diff = 1 / (1 + np.exp(y_pred)) - y

        return (1 / X.shape[0]) * (np.dot(np.transpose(X), diff))

    def _predict(self, X):
        return np.round(self._proba(X))

    def _proba(self, X):
        N, k_num = X.shape
        assert k_num == self.coef.shape[0]

        return self._sigmoid(np.dot(X, self.coef))

    def predict(self, X):
        return np.round(self.proba(X))

    def proba(self, X):
        N, k_num = X.shape
        assert k_num == self.coef.shape[0] - 1

        normalized_X = (X - self.mean) / self.std

        return self._sigmoid(np.dot(GeneralModel._add_intercept(normalized_X), self.coef))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
