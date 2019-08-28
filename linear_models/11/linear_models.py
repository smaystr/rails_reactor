import numpy as np
import matplotlib.pyplot as plt
import metrics


class linear_model():
    def __init__(self, max_iter=int(1e4), lr=1e-2):
        self.max_iter = max_iter
        self.lr = lr

    def fit(self, X, y):
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

    def _gradient_descent(self, X, y):
        cost = []

        for i in range(self.max_iter):

            y_pred = self._predict_yhat(X)
            dW = (X.T @ (y_pred - y)) / y.size

            self._weights -= self.lr * dW

            cost.append(self._cost(self._weights, X, y))


        plt.scatter(list(range(self.max_iter)), cost)
        plt.show()


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
        return np.mean(-y.T @ np.log(y_pred) - (1-y).T @ np.log(1 - y_pred))

    def _predict_yhat(self, X):
        z = X @ self._weights
        return self._sigmoid(z)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def score(self, X, y):
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
        return (1/2*m) * np.sum(np.square(predictions-y))

    def _predict_yhat(self, X):
        return X @ self._weights

    def score(self, X, y):
        y_pred = self.predict(X)

        return metrics.mean_squared_error(y, y_pred)


class StandardScaler():
    def __init__(self):
        self.var_ = 1
        self.mean_ = 0

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.var_ = X.std(axis=0)

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.var_

    def fit_transform(self, X: np.ndarray):
        self.fit(X)

        return self.transform(X)
