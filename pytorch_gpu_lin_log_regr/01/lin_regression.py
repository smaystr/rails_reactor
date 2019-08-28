import torch
import numpy as np

import metrics
import tensorboardX

torch.manual_seed(42)


class LinearRegression():

    def __init__(
        self, learning_rate, max_iter,
        batch, regulization,
        device, C=1
    ):
        self.lr = learning_rate
        self.epoch = max_iter
        self.batch_size = batch
        self.reg = regulization
        self.C = C
        self.device = device

    def _loss(self, y_hat, y):
        rglz_for_l1 = lambda theta: self.C * torch.mean(torch.abs(theta))
        rglz_for_l2 = lambda theta: self.C * torch.mean(theta ** 2)

        rglz = 0
        if self.reg.lower() == 'l1':
            rglz = rglz_for_l1(self.theta)
        elif self.reg.lower() == 'l2':
            rglz = rglz_for_l2(self.theta)
        elif self.reg.lower() == 'l1_l2':
            rglz = rglz_for_l1(self.theta) + rglz_for_l2(self.theta)

        return torch.mean(torch.pow(y_hat - y, 2)) + rglz

    def fit(self, X, y):
        self.theta = torch.rand(X.size()[1] + 1).to(self.device)
        X = torch.cat([X.new_ones(X.size()[0], 1), X], dim=1)

        writer = tensorboardX.SummaryWriter()

        # SGD
        rglz = 0
        for i in range(self.epoch):

            for batch in range(0, X.size()[0], self.batch_size):
                X_batch, y_batch = (
                    X[batch: batch + self.batch_size],
                    y[batch: batch + self.batch_size],
                )

                y_hat = X_batch.mv(self.theta)
                grad = X_batch.t().mv(y_hat - y_batch)

                # regulization
                if self.reg.lower() == 'l1':
                    rglz = self.C * torch.sign(self.theta)
                elif self.reg.lower() == 'l2':
                    rglz = self.C * self.theta
                elif self.reg.lower() == 'l1_l2':
                    rglz = self.C * torch.sign(self.theta) + self.C * self.theta

                self.theta -= self.lr * (grad + rglz) / X_batch.size()[0]

            y_pred = X.mv(self.theta)

            writer.add_scalar('Loss', self._loss(y, y_pred).item(), i)
            writer.add_scalar("RMSE", float(metrics.rmse(y, y_pred).item()), i)
            writer.add_scalar("MSE", float(metrics.mse(y, y_pred).item()), i)
            writer.add_scalar("MAE", float(metrics.mae(y, y_pred).item()), i)

        writer.close()
        return self

    def predict(self, X):
        X = torch.cat([X.new_ones(X.size()[0], 1), X], dim=1)
        return X.mv(self.theta)

    def get_theta(self):
        return self.theta[1:]  # not including bias

    def get_loss(self):
        return self.loss


class LinearRegressionTorch(torch.nn.Module):

    def __init__(self, input_dim, device):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1).to(device)

    def forward(self, X):
        out = self.linear(X)
        return out

    def predict(self, X):
        self.eval()
        return (
            super().__call__(X.to(self.device))
        ).squeeze(dim=1).type(torch.float32)
