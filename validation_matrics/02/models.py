import numpy as np


class LogRegression():

    def __init__(
        self, learning_rate, max_iter,
        regulization, C=1, verbose=False
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.reg = regulization
        self.C = C

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        self.theta = np.random.random(X.shape[1])

        self.loss = dict()

        if self.verbose:
            print('Loss:')

        rglz = 0
        for i in range(self.max_iter):
            z = X @ self.theta
            h = self._sigmoid(z)
            grad = X.T @ (h - y)

            # regulization
            if self.reg == 'L1':
                rglz = self.C * np.sign(self.theta)
            elif self.reg == 'L2':
                rglz = self.C * self.theta
            elif self.reg == 'L1_L2':
                rglz = self.C * np.sign(self.theta) + self.C * self.theta

            self.theta -= self.lr * (grad + rglz) / y.size

            if i % 10000 == 0:
                self.loss[f'iter_{i}'] = self._loss(h, y)
                if self.verbose:
                    print(self.loss[f'iter_{i}'])

        return self

    def predict(self, X):
        return self._sigmoid(X @ self.theta) >= 0.5

    def predict_proba(self, X):
        return self._sigmoid(X @ self.theta)

    def score(self, X_test, y_test, metric):
        return metric(self.predict(X_test), y_test)

    def get_theta(self):
        return self.theta

    def get_loss(self):
        return self.loss


class LinearRegressor():

    def __init__(
        self, learning_rate, max_iter,
        regulization, C=1, verbose=False
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.reg = regulization
        self.C = C

    def _loss(self, y_hat, y):
        return np.sqrt(np.mean(np.square(y_hat - y)))

    def fit(self, X, y):
        self.theta = np.random.random((X.shape[1] + 1, 1))

        y = y.reshape((-1, 1))
        X_ = np.ones((X.shape[0], X.shape[1] + 1))
        X_[:, 1:] = X

        self.loss = dict()

        if self.verbose:
            print('Loss:')

        rglz = 0
        for i in range(self.max_iter):
            y_hat = X_.dot(self.theta)
            grad = X_.T @ (y_hat - y)

            # regulization
            if self.reg == 'L1':
                rglz = self.C * np.sign(self.theta)
            elif self.reg == 'L2':
                rglz = self.C * self.theta
            elif self.reg == 'L1_L2':
                rglz = self.C * np.sign(self.theta) + self.C * self.theta

            self.theta -= self.lr * (grad + rglz) / X_.shape[0]

            if i % 10000 == 0:
                self.loss[f'iter_{i}'] = self._loss(y_hat, y)
                if self.verbose:
                    print(self.loss[f'iter_{i}'])

        return self

    def predict(self, X):
        X_ = np.ones((X.shape[0], X.shape[1] + 1))
        X_[:, 1:] = X
        return X_.dot(self.theta)

    def score(self, X, y, metric):
        return metric(self.predict(X), y.reshape((-1, 1)))

    def get_theta(self):
        return [w[0] for w in self.theta[1:]]  # not including bias

    def get_loss(self):
        return self.loss
