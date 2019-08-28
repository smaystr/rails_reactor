import numpy as np
from metrics import accuracy, precision, recall, f1, mse, rmse, mae


def _add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


class LogisticRegression:
    # fit_intercept is True by default and isn't tunable, because it will lower over metrics a lot
    def __init__(self, lr=1e-4, num_iter=1000, penalty='l2', C=1.0, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.penalty = penalty
        self.C = C
        self.verbose = verbose

        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        self.weights = np.zeros((X.shape[1] + 1, 1), dtype=np.float32)
        y_true = y.reshape((-1, 1))
        X_ =_add_intercept(X)

        for i in range(self.num_iter):
            reg_p = 0
            predicted = self.predict_proba(X)
            grad = X_.T.dot(predicted - y_true)

            if self.penalty == 'l1':
                reg_p = self.C * np.sign(self.weights)
            elif self.penalty == 'l2':
                reg_p = self.C * self.weights

            self.weights -= self.lr * (grad + reg_p) / X.shape[0]

            if self.verbose and i % 10000 == 0:
                z = np.dot(X_, self.weights)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y_true)} \t')

    def predict_proba(self, X):
        X = _add_intercept(X)
        return self.__sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=.55):
        return self.predict_proba(X) >= threshold

    def score(self, X, y, thr=.5, metric='accuracy'):
        return self.metrics.get(metric, 'accuracy')(self.predict(X, thr), y.reshape(-1, 1))


class LinearRegression:
    def __init__(self, lr=1e-4, num_iter=100, penalty='l2', C=1.0):
        self.lr = lr
        self.num_iter = num_iter
        self.penalty = penalty
        self.C = C

        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

    def fit(self, X, y):
        self.theta = np.zeros((X.shape[1] + 1, 1), dtype=np.float32)
        y_true = y.reshape((-1, 1))
        X_ = _add_intercept(X)

        for i in range(self.num_iter):
            reg_p = 0
            predicted = self.predict(X)
            grad = X_.T.dot(predicted - y_true)

            if self.penalty == 'l1':
                reg_p = self.C * np.sign(self.theta)
            elif self.penalty == 'l2':
                reg_p = self.C * self.theta

            self.theta -= self.lr * (grad + reg_p) / X.shape[0]

    def predict(self, X):
        X_ = _add_intercept(X)
        return X_.dot(self.theta)

    def get_score(self, X, y, metric='rmse'):
        return self.metrics.get(metric, 'rmse')(self.predict(X), y.reshape((-1, 1)))
