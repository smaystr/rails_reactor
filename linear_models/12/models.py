import numpy as np
import metrics


class LogisticRegression():

    def __init__(self, lr=1e-5, epoch=100, threshold=0.5):
        self.lr = lr
        self.epoch = epoch
        self.threshold = threshold

    def fit(self, X, Y):
        self.w = np.zeros((X.shape[1] + 1, 1), dtype=np.float32)
        m = X.shape[0]
        y = Y.reshape((-1, 1))
        x = np.ones((X.shape[0], X.shape[1] + 1))
        x[:, 1:] = X

        for i in range(self.epoch):
            predicted = self.predict_proba(X)
            dw = np.dot(x.T, predicted - y) / m

            self.w = self.w - self.lr * dw

        return self

    def predict(self, X):
        return np.where(self.predict_proba(X) >= self.threshold, 1, 0)

    def predict_proba(self, X):
        x = np.ones((X.shape[0], X.shape[1] + 1))
        x[:, 1:] = X
        z = x.dot(self.w)
        return 1 / (1 + np.exp(-z))

    def score(self, X, Y, metric):
        y = Y.reshape((-1, 1))
        return metric(self.predict(X), y)


class LinearRegression():

    def __init__(self, lr=1e-5, epoch=10000):
        self.lr = lr
        self.epoch = epoch

    def fit(self, X, Y):
        self.w = np.zeros((X.shape[1] + 1, 1), dtype=np.float32)
        m = X.shape[0]
        y = Y.reshape((-1, 1))
        x = np.ones((X.shape[0], X.shape[1] + 1))
        x[:, 1:] = X
        print(y.dtype)
        for i in range(self.epoch):
            predicted = self.predict(X)
            dw = np.dot(x.T, predicted - y) / m
            if i % 50 == 0:
                print(self.score(X, Y, metric=metrics.rmse))

            self.w = self.w - self.lr * dw

        return self

    def predict(self, X):
        x = np.ones((X.shape[0], X.shape[1] + 1))
        x[:, 1:] = X
        return x.dot(self.w)

    def score(self, X, Y, metric):
        y = Y.reshape((-1, 1))
        return metric(self.predict(X), y)
