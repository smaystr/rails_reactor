import logging
from abc import abstractmethod

import torch
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class BaseModel(BaseEstimator):
    def __init__(
        self,
        device: torch.device = torch.device('cpu'),
        learning_rate: float = 0.01,
        num_iterations: int = 10000,
        C=0.1,
        fit_intercept: bool = True,
        batch_size: int = 32
    ):
        self.learning_rate = learning_rate
        self.num_iter = num_iterations
        self.C = C
        self.fit_intercept = fit_intercept
        self.device = device
        self.batch_size = batch_size
        self.weights = None

    def fit(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)

        logging.info('Fitting logreg')

        if self.fit_intercept:
            X = self._add_intercept(X)

        # weights initialization
        self.weights = torch.zeros((X.shape[1], 1), dtype=torch.float32, device=self.device)

        data_size = X.shape[0]

        for i in range(self.num_iter):
            for j in range(data_size // self.batch_size):
                X_batch = X[j * self.batch_size: j * self.batch_size + self.batch_size]
                y_batch = y[j * self.batch_size: j * self.batch_size + self.batch_size]

                self.weights -= (self.learning_rate / data_size) * self._step(X_batch, y_batch)

        return self

    def _add_intercept(self, X):
        return torch.cat([torch.ones((X.shape[0], 1), dtype=torch.float32, device=self.device), X], dim=1)

    @abstractmethod
    def _loss(self, logit, y):
        raise NotImplementedError

    @abstractmethod
    def _step(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError


class LinearRegression(RegressorMixin, BaseModel):
    def _loss(self, h, y):
        return ((y - h).mm(y - h)).mean() + self.C * (self.weights.mm(self.weights)).mean()

    def _step(self, X, y):
        return X.t().mm(X.mm(self.weights) - y) + self.C * self.weights

    def predict(self, x):
        x = x.to(self.device)
        if self.fit_intercept:
            x = self._add_intercept(x)

        return torch.mm(x, self.weights)


class LogisticRegression(ClassifierMixin, BaseModel):
    def _loss(self, h, y):
        return ((-y * torch.log(h) - (1 - y) * torch.log(1 - h)) + self.C * torch.sum(
            self.weights.mm(self.weights))).mean()

    def _step(self, X, y):
        return X.t().mm(self._sigmoid(X.mm(self.weights)) - y) + self.C * self.weights

    def _sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def predict_prob(self, x):
        if self.fit_intercept:
            x = self._add_intercept(x)

        return self._sigmoid(x.mm(self.weights))

    def predict(self, x):
        x = x.to(self.device)
        return self.predict_prob(x).round()
