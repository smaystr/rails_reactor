import torch
import time
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


class LinearModel:
    def __init__(self, epochs=int(1e4), lr=1e-2, batch_size=10, device='cpu', C=1.0, penalty=None):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        self.penalty = penalty
        if penalty == 'l1':
            self.penalty_func = self._lasso
            self.penalty_dt = self._lasso_dt
        elif penalty == 'l2':
            self.penalty_func = self._ridge
            self.penalty_dt = self._ridge_dt
        else:
            self.penalty_func = lambda weights: 0
            self.penalty_dt = lambda weights: torch.zeros(weights.shape)

        self.C = C

    def _ridge(self, weights):
        weights = weights.copy()
        weights[0] = 0
        return 1/self.C * 1/2 * torch.sum(weights**2)

    def _ridge_dt(self, weights):
        weights = weights.copy()
        weights[0] = 0
        return 1/self.C * weights

    def _lasso(self, weights):
        weights = weights.copy()
        weights[0] = 0
        return 1/self.C * torch.sum(torch.abs(weights))

    def _lasso_dt(self, weights):
        weights = weights.copy()
        weights[0] = 0
        return 1/self.C * torch.sign(weights)

    def fit(self, inputs, labels):
        inputs = inputs.to(self.device)

        start_time = time.time()

        labels = labels.reshape(-1, 1).to(self.device)

        n_counts, n_features = inputs.shape

        # add intercept column
        inputs_extended = torch.ones((n_counts, n_features + 1))
        inputs_extended[:, 1:] = inputs
        inputs_extended = inputs_extended.double().to(self.device)

        self._weights = torch.rand(n_features+1, 1).double().to(self.device)
        self._gradient_descent(inputs_extended, labels)

        self.intercept_ = self._weights[0, :]
        self.coef_ = self._weights[1:, :]

        self.score_ = self.score(inputs, labels)

        self.time = time.time() - start_time

    def _gradient_descent(self, X, y):
        cost = []
        n_counts = X.shape[0]

        for i in range(int(self.epochs)):
            np.random.shuffle(X)

            for i in range(self.batch_size, n_counts+1, self.batch_size):
                X_batch, y_batch = X[i-self.batch_size: i].to(
                    self.device), y[i-self.batch_size: i].to(self.device)
                y_pred = self._predict_yhat(X_batch)
                dW = (torch.transpose(X_batch, 0, 1) @ (y_pred - y_batch)) / y_batch.size()[0] + self.penalty_dt(self._weights).double().to(self.device)

                self._weights -= self.lr * dW

            # if there were elements that left after mini-batches:
            # i saved last index from which elements were not taken
            if n_counts % self.batch_size != 0:
                X_batch, y_batch = X[i: ].to(
                    self.device), y[i: ].to(self.device)
                y_pred = self._predict_yhat(X_batch)
                dW = (torch.transpose(X_batch, 0, 1) @ (y_pred - y_batch)) / y_batch.size()[0] + self.penalty_dt(self._weights).double().to(self.device)

                self._weights -= self.lr * dW

            cost.append(self._cost(self._weights, X, y))

        self.cost = cost[-1]


class LogisticRegression(LinearModel):
    def __init__(self, epochs, lr, batch_size, device, threshold=0.5):
        super().__init__(epochs, lr, batch_size, device)
        self.threshold = threshold

    def predict(self, X):
        X = X.to(self.device)
        h = self._sigmoid(torch.mm(X, self.coef_) + self.intercept_)

        prediction = (h > self.threshold).int()

        return prediction

    def _cost(self, weights, X, y):
        y_pred = self._sigmoid(torch.mm(X, weights))
        return torch.mean(-torch.transpose(y, 0, 1) @ torch.log(y_pred) - torch.transpose(1-y, 0, 1) @ torch.log(1 - y_pred)
                          + self.penalty_func(weights))

    def _predict_yhat(self, X):
        z = torch.mm(X,  self._weights)
        return self._sigmoid(z)

    def _sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def score(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        self.best_is_max = True

        y_pred = self.predict(X)

        return accuracy_score(y.cpu(), y_pred.cpu())


class LinearRegression(LinearModel):

    def __init__(self, epochs, lr, batch_size, device):
        super().__init__(epochs, lr, batch_size, device)

    def predict(self, X):
        X = X.to(self.device)
        return torch.mm(X, self.coef_) + self.intercept_

    def _cost(self, weights, X, y):
        n_counts = y.size()[0]

        predictions = torch.mm(X, weights)
        return (1/2*n_counts) * (torch.sum(torch.pow(predictions-y, 2)) + self.penalty_func(weights))

    def _predict_yhat(self, X):
        return torch.mm(X, self._weights)

    def score(self, X, y):
        self.best_is_max = False
        y_pred = self.predict(X).cpu()
        y = y.cpu()

        return mean_squared_error(y, y_pred)
