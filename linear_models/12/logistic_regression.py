import numpy as np
import metrics


class LogisticRegression():

    def __init__(self, lr=1e-5, epoch=100, threshold=0.5,  regularization='L2', alpha=1):
        self.lr = lr
        self.epoch = epoch
        self.threshold = threshold
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

<<<<<<< HEAD
            self.w = self.w - self.lr * weights_derivative
=======
            self.w = self.w - self.lr * weights_derivative 
>>>>>>> aa8030b0394659a07465db083af02786b0d8399f

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

