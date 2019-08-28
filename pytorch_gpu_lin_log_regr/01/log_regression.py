import torch
import numpy as np
import metrics

import tensorboardX

torch.manual_seed(42)


class LogRegression():

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

    def _sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def _loss(self, y_hat, y):
        return torch.mean(
            -y * torch.log(y_hat) - (1 - y) * torch.log(1 - y_hat)
        )

    def fit(self, X, y):
        self.theta = torch.rand(X.size()[1]).to(self.device)

        writer = tensorboardX.SummaryWriter()

        # SGD
        rglz = 0
        for i in range(self.epoch):

            for batch in range(0, X.size()[0], self.batch_size):
                X_batch, y_batch = (
                    X[batch: batch + self.batch_size],
                    y[batch: batch + self.batch_size],
                )

                y_hat = self._sigmoid(X_batch.mv(self.theta))
                grad = X_batch.t().mv(y_hat - y_batch)

                # regulization
                if self.reg.lower() == 'l1':
                    rglz = self.C * torch.sign(self.theta)
                elif self.reg.lower() == 'l2':
                    rglz = self.C * self.theta
                elif self.reg.lower() == 'l1_l2':
                    rglz = self.C * torch.sign(self.theta) + self.C * self.theta

                self.theta -= self.lr * (grad + rglz) / X_batch.size()[0]

            y_pred = self.predict(X)

            writer.add_scalar('Loss', sefl._loss(y, y_pred).item(), i)
            writer.add_scalar("Accuracy", float(metrics.accuracy(y, y_pred).item()), i)
            writer.add_scalar("Recall", float(metrics.recall(y, y_pred).item()), i)
            writer.add_scalar("Precision", float(metrics.precision(y, y_pred).item()), i)
            writer.add_scalar("F1", float(metrics.f1(y, y_pred).item()), i)

        writer.close()
        return self

    def predict(self, X):
        return (self._sigmoid(X.mv(self.theta)) >= 0.5).type(torch.float32)

    def predict_proba(self, X):
        return self._sigmoid(X.mv(self.theta))

    def get_theta(self):
        return self.theta

    def get_loss(self):
        return self.loss


class LogRegressionTorch(torch.nn.Module):

    def __init__(self, input_dim, device):
        super().__init__()
        self.device = device
        self.linear = torch.nn.Linear(input_dim, 1).to(device)

    def forward(self, X):
        out = self.linear(X)
        return torch.sigmoid(out)

    def predict(self, X):
        self.eval()
        return (
            super().__call__(X.to(self.device)) > 0.5
        ).squeeze(dim=1).type(torch.float32)
