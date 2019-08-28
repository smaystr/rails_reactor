import numpy as np
import metrics


class LinearRegression():

    def __init__(self, lr=1e-5, epoch=10000, regularization='L2', alpha=1):
        self.lr = lr
        self.epoch = epoch
        self.regularization = self.compute_regularization(regularization)
        self.alpha = alpha

    def compute_regularization(self, reg):
        if reg == 'L1':
            return self.compute_l1()
        elif reg == 'L2':
            return self.compute_l2()
        else:
            return self.return_0()

    def compute_l1(self):
        return np.sign(self.w) * self.alpha

    def compute_l2(self):
        return self.w

    def return_0(self):
        return 0

    def fit(self, X, Y):
        self.w = np.zeros((X.shape[1] + 1, 1), dtype=np.float32)
        n_samples = X.shape[0]
        y = Y.reshape((-1, 1))
        x = np.ones((X.shape[0], X.shape[1] + 1))
        x[:, 1:] = X

        for i in range(self.epoch):
            predicted = self.predict_proba(X)
            weights_derivative = (np.dot(x.T, predicted - y) + self.regularization()) / n_samples

            self.w = self.w - self.lr * weights_derivative

        return self

    def predict(self, X):
        x = np.ones((X.shape[0], X.shape[1] + 1))
        x[:, 1:] = X
        return x.dot(self.w)

    def score(self, X, Y, metric):
        y = Y.reshape((-1, 1))
        return metric(self.predict(X), y)
