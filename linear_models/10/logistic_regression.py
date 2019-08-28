import numpy as np


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
    ):
        self.lr = lr
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eps = eps
        self.verbose_rounds = verbose_rounds
        self.not_training = True
        self.tol = tol

        penalties = set(["l1", "l2"])
        penalty = penalty.lower()

        if penalty in penalties:
            self.penalty = penalty
        else:
            raise Exception(
                f"Penalty should be {penalties}. The the penalty was: {penalty}"
            )

    def init_weights(self, cols):
        self.weights = np.ones((cols, 1))

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
        self.not_training = False

        self.init_weights(X.shape[1])

        change = 0
        for i in range(self.max_iter):
            objective = self.predict_proba(X)

            gradient = (1 / self.size) * np.dot(X.T, objective - y) + self.regularize()
            self.weights -= self.lr * gradient

            change = np.abs(np.sum(objective - self.predict_proba(X)))
            if i % self.verbose_rounds == 0:
                print(f"Loss after {i} steps: {self.loss(X,y)}")

            if change < self.tol:
                break

        self.not_training = True

    def predict_proba(self, X):
        X = self.add_intercept(X)
        return 1 / (1 + np.exp(-1 * np.dot(X, self.weights) + self.eps))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype("uint8").ravel()
