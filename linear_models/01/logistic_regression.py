import numpy as np
from utils import sigmoid, log_loss, accuracy, prepare_data


class LogisticRegression:

    def __init__(self, tol=0.0001, fit_intercept=True, max_iter=10000):
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, x_train, y_train, lr=0.0001):
        x_train, y_train = prepare_data(x_train, y_train, self.fit_intercept)
        self.weights = np.zeros(x_train.shape[1])
        pred = self.predict_proba(x_train)
        loss = log_loss(y_train, pred)
        for i in range(self.max_iter):
            self.weights -= lr * np.dot(x_train.T, pred - y_train) / x_train.shape[0]  # gradient descent
            pred = self.predict_proba(x_train)
            new_loss = log_loss(y_train, pred)
            if abs(loss - new_loss) < self.tol:
                break
            loss = new_loss
        if self.fit_intercept:
            self.coef_ = self.weights[1:]
            self.intercept_ = self.weights[0]
        else:
            self.coef_ = self.weights
            self.intercept_ = .0
        return self

    def predict_proba(self, x):
        return sigmoid(np.dot(x, self.weights))

    def predict(self, x):
        classify = self.predict_proba(x) > 0.5
        return classify.astype(int)

    def score(self, x_test, y_test):
        x_test, y_test = prepare_data(x_test, y_test, self.fit_intercept)
        pred = self.predict(x_test)
        return accuracy(y_test, pred)
