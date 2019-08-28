import numpy as np


class LinearRegression:
    def __init__(
        self,
        lr=0.0001,
        mode="grad",
        tol=1e-3,
        fit_intercept=True,
        verbose_rounds=200,
        verbose=False,
        max_iter=200,
    ):
        self.fit_intercept = fit_intercept
        self.not_training = True
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.verbose_rounds = verbose_rounds
        self.verbose = verbose

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

    def init_weights(self):
        self.weights = np.random.normal(size=(self.cols, 1))

    def cost(self, X, y):
        return (np.square(np.dot(X, self.weights) - y)) / (2 * X.shape[0])

    def fit(self, X, y):
        self.size = len(X)
        X = self.add_intercept(X)
        self.cols = X.shape[1]
        self.not_training = False

        y = np.reshape(y, (y.shape[0], 1))
        if self.mode == "grad":
            self.init_weights()

            change = 0
            for i in range(int(self.max_iter)):

                objective = self.cost(X, y)
                self.weights -= (
                    self.lr / X.shape[0] * np.dot(X.T, np.dot(X, self.weights) - y)
                )

                new_obj = self.cost(X, y)
                change = np.abs(np.sum(objective - new_obj))

                if self.verbose and i % self.verbose_rounds == 0:
                    print(f"Error at round {i}: {np.sum(new_obj)}")

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
        return np.dot(X, self.weights).reshape((len(X), 1))


class LogisticRegression:
    def __init__(
        self,
        lr=0.001,
        C=1,
        max_iter=250,
        fit_intercept=True,
        penalty="l2",
        eps=1e-12,
        verbose_rounds=200,
        tol=1e-5,
        verbose=False,
        threshold=0.5,
    ):
        self.threshold = threshold
        self.lr = lr
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eps = eps
        self.verbose_rounds = verbose_rounds
        self.not_training = True
        self.tol = tol
        self.verbose = verbose

        penalties = set(["l1", "l2"])
        penalty = penalty.lower()

        if penalty in penalties:
            self.penalty = penalty
        else:
            raise Exception(
                f"Penalty should be {penalties}. The the penalty was: {penalty}"
            )

    def init_weights(self):
        self.weights = np.ones((self.cols, 1))

    def loss(self, X, y):
        sigm = self.predict_proba(X)
        return (-y * np.log(sigm) - (1 - y) * np.log(1 - sigm)).mean()

    def add_intercept(self, X):
        if self.fit_intercept and self.not_training:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X

    def regularize(self):
        if self.penalty == "l1":
            return self.weights / (self.C * self.size * np.abs(self.weights))
        return 1 / (self.C * self.size) * self.weights

    def fit(self, X, y):
        self.size = len(X)
        X = self.add_intercept(X)
        self.cols = X.shape[1]
        self.not_training = False

        self.init_weights()

        y = np.reshape(y, (len(y), 1))

        change = 0
        for i in range(int(self.max_iter)):
            objective = self.predict_proba(X)

            gradient = (1 / self.size) * np.dot(X.T, objective - y) + self.regularize()
            self.weights -= self.lr * gradient

            change = np.abs(np.sum(objective - self.predict_proba(X)))
            if self.verbose and i % self.verbose_rounds == 0:
                print(f"Loss after {i} steps: {self.loss(X,y)}")

            if change < self.tol:
                break

        self.not_training = True

    def predict_proba(self, X):
        X = self.add_intercept(X)
        return 1 / (1 + np.exp(-1 * np.dot(X, self.weights) + self.eps))

    def predict(self, X):
        return (
            (self.predict_proba(X) > self.threshold)
            .astype("uint8")
            .ravel()
            .reshape((len(X), 1))
        )
