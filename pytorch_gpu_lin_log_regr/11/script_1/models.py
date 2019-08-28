import torch


# from abc import ABC, abstractmethod


class Regressor:
    def __init__(self, alpha=1e-3, iters=100, reg=None, C=1.0, batch_size=32, device='cpu'):
        self.alpha = alpha
        self.iters = iters
        self.reg = reg
        self.C = C
        self.batch_size = batch_size
        self.device = device
        self.coef = None

    def fit(self, X, y):
        X = torch.cat([torch.ones((X.shape[0], 1), dtype=torch.float32, device=self.device), X], dim=1)
        self.coef = torch.zeros((X.shape[1], 1), dtype=torch.float32, device=self.device)
        dataset_size = X.shape[0]

        for i in range(self.iters):
            for j in range(dataset_size // self.batch_size):
                X_batch = X[j * self.batch_size: (j + 1) * self.batch_size]
                y_batch = y[j * self.batch_size: (j + 1) * self.batch_size]
                self.coef -= (self.alpha / dataset_size) * (self.gradient(X_batch, y_batch) + self.reg_param())

        return self

    def reg_param(self):
        if self.reg == 'L1':
            return self.C * torch.sign(self.coef)
        elif self.reg == 'L2':
            return self.C * self.coef
        else:
            return 0.0

    def predict(self, X):
        return torch.cat([torch.ones((X.shape[0], 1), dtype=torch.float32, device=self.device), X], dim=1).mm(self.coef)

    def gradient(self, X, y):
        pass


class LinearRegression(Regressor):

    def __init__(self, alpha=1e-1, iters=100, reg=None, C=1.0, batch_size=32, device='cpu'):
        super().__init__(alpha, iters, reg, C, batch_size, device)

    def gradient(self, X, y):
        return X.t().mm(X.mm(self.coef) - y)


class LogisticRegression(Regressor):

    def __init__(self, alpha=1e-1, iters=100, reg=None, C=1.0, batch_size=32, device='cpu'):
        super().__init__(alpha, iters, reg, C, batch_size, device)

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def predict(self, X):
        return self.sigmoid(Regressor.predict(self, X)).round()

    def gradient(self, X, y):
        return X.t().mm(self.sigmoid(X.mm(self.coef)) - y)
