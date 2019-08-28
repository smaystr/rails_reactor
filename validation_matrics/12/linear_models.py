import numpy as np
import metrics
import time


class linear_model():
    def __init__(self, max_iter=int(1e4), lr=1e-2, C=1.0, penalty=None):
        self.max_iter = max_iter
        self.lr = lr
        self.penalty = penalty
        if penalty == 'l1':
            self.penalty_func = self._lasso
            self.penalty_dt = self._lasso_dt
        elif penalty == 'l2':
            self.penalty_func = self._ridge
            self.penalty_dt = self._ridge_dt
        else:
            self.penalty_func = lambda x: 0
            self.penalty_dt = lambda weights: np.zeros(weights.shape)

        self.C = C

    def _ridge(self, weights):
        weights = weights.copy()
        weights[0] = 0
        return 1/self.C * 1/2 * np.sum(weights**2)

    def _ridge_dt(self, weights, n):
        weights = weights.copy()
        weights[0] = 0
        return 1/self.C * weights

    def _lasso(self, weights):
        weights = weights.copy()
        weights[0] = 0
        return 1/self.C * np.sum(np.abs(weights))

    def _lasso_dt(self, weights):
        weights = weights.copy()
        weights[0] = 0
        return 1/self.C * np.sign(weights)

    def fit(self, X, y):
        t = time.time()

        y = y.reshape(-1, 1)

        n, n_features = X.shape

        # add intercept column
        X_ = np.ones((n, n_features + 1))
        X_[:, 1:] = X

        self._weights = np.random.rand(n_features+1, 1)

        self._gradient_descent(X_, y)

        self.intercept_ = self._weights[0, :]
        self.coef_ = self._weights[1:, :]

        self.score_ = self.score(X, y)

        self.time = time.time() - t

    def _gradient_descent(self, X, y):
        cost = []
        for i in range(int(self.max_iter)):

            y_pred = self._predict_yhat(X)
            dW = (X.T @ (y_pred - y)) / y.size + self.penalty_dt(self._weights)

            self._weights -= self.lr * dW

            cost.append(self._cost(self._weights, X, y))

        self.cost = cost[-1]


class LogisticRegression(linear_model):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def predict(self, X):
        h = self._sigmoid(np.dot(X, self.coef_) + self.intercept_)

        prediction = (h > self.threshold).astype(int)

        return prediction

    def _cost(self, weights, X, y):
        y_pred = self._sigmoid(np.dot(X, weights))
        return np.mean(-y.T @ np.log(y_pred) - (1-y).T @ np.log(1 - y_pred)
                       + self.penalty_func(weights))

    def _predict_yhat(self, X):
        z = X @ self._weights
        return self._sigmoid(z)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def score(self, X, y):
        self.best_is_max = True

        y_pred = self.predict(X)

        return metrics.accuracy_score(y, y_pred)


class LinearRegression(linear_model):

    def __init__(self):
        super().__init__()

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def _cost(self, weights, X, y):
        m = y.size

        predictions = np.dot(X, weights)
        return (1/2*m) * (np.sum(np.square(predictions-y))
                          + self.penalty_func(weights))

    def _predict_yhat(self, X):
        return X @ self._weights

    def score(self, X, y):
        self.best_is_max = False
        y_pred = self.predict(X)

        return metrics.mean_squared_error(y, y_pred)
