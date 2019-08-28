import numpy as np


class LinearRegression:
    def __init__(
        self,
        lr=0.0001,
        mode="grad",
        tol=1e-3,
        fit_intercept=True,
        verbose_rounds=200,
        max_iter=200,
    ):
        self.fit_intercept = fit_intercept
        self.not_training = True
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.verbose_rounds = verbose_rounds

        modes = set(["pseudoinv", "grad", "normal", "qr"])
        mode = mode.lower()

        if mode in modes:
            self.mode = mode
        else:
            raise Exception(f"You set mode {mode}. Acceptable values are {modes}")

    def add_intercept(self, X):
        if self.fit_intercept and self.not_training:

            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X

    def init_weights(self, cols):
        self.weights = np.random.normal(size=(cols, 1))

    def cost(self, X, y):
        return (np.square(np.dot(X, self.weights) - y)) / (2 * X.shape[0])

    def fit(self, X, y):
        self.size = len(X)

        X = self.add_intercept(X)
        self.not_training = False

        if self.mode == "grad":
            self.init_weights(X.shape[1])

            change = 0
            for i in range(self.max_iter):
                objective = self.cost(X, y)
                self.weights -= (
                    self.lr / X.shape[0] * np.dot(X.T, np.dot(X, self.weights) - y)
                )

                new_error = self.cost(X, y)
                change = np.abs(np.sum(objective - new_error))

                if i % self.verbose_rounds == 0:
                    print(f"Error at round {i}: {np.sum(new_error)}")

                if change < self.tol:
                    break

        if self.mode == "normal":
            det = np.linalg.det(np.dot(X.T, X))
            if det == 0:
                raise Exception("Can't compute normal equation on singular X.T*X")

            self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
            print(f"Error with normal equation: {np.sum(self.cost(X,y))}")

        if self.mode == "pseudoinv":
            self.weights = np.dot(np.linalg.pinv(X), y)
            print(f"Error with pseudoinverse: {np.sum(self.cost(X,y))}")

        if self.mode == "qr":
            Q, R = np.linalg.qr(X)
            self.weights = np.dot(np.linalg.inv(R), np.dot(Q.T, y))
            print(f"Error with QR decomposition: {np.sum(self.cost(X,y))}")

        self.not_training = True

    def evaluate(self, X, y):
        X = self.add_intercept(X)
        return np.sum((np.square(np.dot(X, self.weights) - y)) / (X.shape[0]))

    def predict(self, X):
        X = self.add_intercept(X)
        return np.dot(X, self.weights)
