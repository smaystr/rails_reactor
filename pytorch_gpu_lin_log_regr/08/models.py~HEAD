from time import time
import numpy as np
import torch
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter
from metrics import accuracy, precision, recall, f1, mse, rmse, mae


def _add_intercept(X):
    X_ = np.ones((X.shape[0], X.shape[1] + 1))
    X_[:, 1:] = X
    return torch.tensor(X_, dtype=torch.float)


class LogisticRegression:
    # fit_intercept is True by default and isn't tunable, because it will lower over metrics a lot
    def __init__(self, lr=1e-4, batch=None, num_iter=1000, penalty='l2', C=1.0, is_cuda=False, verbose=False):
        self.lr = lr
        self.batch = batch
        self.num_iter = num_iter
        self.penalty = penalty
        self.C = C
        self.is_cuda = is_cuda
        self.verbose = verbose

        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        self.writer = SummaryWriter(log_dir=f'logs/low_level_logit')

    def __sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def fit(self, X, y):
        self.weights = torch.zeros((X.shape[1] + 1, 1), dtype=torch.float)
        if self.is_cuda:
            self.weights = self.weights.cuda()
        y_true = torch.tensor(y.reshape((-1, 1)), dtype=torch.float)
        X_ = _add_intercept(X)

        if self.batch:
            step = int(np.ceil(X.shape[0] / self.batch))
        else:
            step = 1

        start_time = time()
        indices = np.arange(X.shape[0])
        for i in range(self.num_iter):
            for j in range(step):
                X_step = X_[indices[j * self.batch: (j + 1) * self.batch]]
                y_step = y_true[indices[j * self.batch: (j + 1) * self.batch]]
                predicted = self.predict_proba(X_step, False)
                grad = X_step.transpose(0, 1).matmul(predicted - y_step)

                reg_p = 0
                if self.penalty == 'l1':
                    reg_p = self.C * self.weights.sign()
                elif self.penalty == 'l2':
                    reg_p = self.C * torch.pow(2, self.weights)

                self.weights -= self.lr * (grad + reg_p) / self.batch

            self.writer.add_scalar('loss', self.loss(X, y), i)
            for k in self.metrics.keys():
                self.writer.add_scalar(k, self.score(X, y, metric=k), i)

            if self.verbose and i % 100 == 0:
                print(f'loss: {self.loss(X, y)} \t')
        print(f'Low level logit trained in {time() - start_time}')

    def predict_proba(self, X, intercept=True):
        if intercept:
            X = _add_intercept(X)
        return self.__sigmoid(X.matmul(self.weights))

    def predict(self, X, thr=.5):
        return self.predict_proba(X) > thr

    def score(self, X, y, metric='accuracy'):
        return self.metrics.get(metric, 'accuracy')(self.predict(X), torch.tensor(y.reshape(-1, 1), dtype=torch.uint8))

    def loss(self, X, y):
        y_true = torch.tensor(y.reshape((-1, 1)), dtype=torch.float)
        y_pred = self.predict_proba(X).reshape((-1, 1))
        return -torch.mean(((1 - y_true) * (1 - y_pred).log()) + (y_true * y_pred.log()))


class LinearRegression:
    def __init__(self, lr=1e-4, batch=None, num_iter=100, penalty='l2', is_cuda=False, C=1.0):
        self.lr = lr
        self.batch = batch
        self.num_iter = num_iter
        self.penalty = penalty
        self.is_cuda = is_cuda
        self.C = C

        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        self.writer = SummaryWriter(log_dir=f'logs/low_level_linreg')

    def fit(self, X, y):
        self.weights = torch.zeros((X.shape[1] + 1, 1), dtype=torch.float)
        if self.is_cuda:
            self.weights = self.weights.cuda()
        y_true = torch.tensor(y.reshape((-1, 1)), dtype=torch.float)
        X_ = _add_intercept(X)

        if self.batch:
            step = int(np.ceil(X.shape[0] / self.batch))
        else:
            step = 1

        start_time = time()
        indices = np.arange(X.shape[0])
        for i in range(self.num_iter):
            for j in range(step):
                X_step = X_[indices[j * self.batch: (j + 1) * self.batch]]
                y_step = y_true[indices[j * self.batch: (j + 1) * self.batch]]
                predicted = self.predict(X_step, False)
                grad = X_step.transpose(0, 1).matmul(predicted - y_step)

                reg_p = 0
                if self.penalty == 'l1':
                    reg_p = self.C * self.weights.sign()
                elif self.penalty == 'l2':
                    reg_p = self.C * torch.pow(2, self.weights)

                self.weights -= self.lr * (grad + reg_p) / self.batch
            self.writer.add_scalar('loss', self.loss(X, y), i)
            for k in self.metrics.keys():
                self.writer.add_scalar(k, self.score(X, y, metric=k), i)

        print(f'Low level linreg trained in {time() - start_time}')

    def predict(self, X, intercept=True):
        if intercept:
            X = _add_intercept(X)
        return X.matmul(self.weights)

    def loss(self, X, y):
        y_true = torch.tensor(y.reshape((-1, 1)), dtype=torch.float)
        y_pred = self.predict(X).reshape((-1, 1))
        return torch.mean((y_true - y_pred).pow(2)) / 2

    def score(self, X, y, metric='rmse'):
        return self.metrics.get(metric, 'rmse')(self.predict(X), torch.tensor(y.reshape((-1, 1)), dtype=torch.float))


class LogitTorch(torch.nn.Module):
    def __init__(self, input_dim, cuda=False):
        super(LogitTorch, self).__init__()
        self.linear = Linear(input_dim, 1)
        if cuda:
            self.linear = self.linear.cuda()

    def forward(self, X):
        output = self.linear(X)
        return torch.sigmoid(output)


class LinRegTorch(torch.nn.Module):
    def __init__(self, input_dim, cuda=False):
        super(LinRegTorch, self).__init__()
        self.linear = Linear(input_dim, 1)
        if cuda:
            self.linear = self.linear.cuda()

    def forward(self, X):
        output = self.linear(X)
        return output


class LinearTrainer:
    def __init__(self, model, lr, epoch, batch):
        self.model = model
        self.epoch = epoch
        self.batch = batch
        if isinstance(model, LinRegTorch):
            self.criterion = torch.nn.MSELoss()
            model_name = 'high_level_linreg'
            self.metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
        elif isinstance(model, LogitTorch):
            self.criterion = torch.nn.BCELoss()
            model_name = 'high_level_logit'
            self.metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        else:
            raise RuntimeError('Undefined model. Pass LogitTorch or LinRegTorch')
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir=f'logs/{model_name}')

    def train(self, X, y):
        X_ = torch.tensor(X, dtype=torch.float)
        y_true = torch.tensor(y.reshape((-1, 1)), dtype=torch.float)
        step = int(np.ceil(X.shape[0] / self.batch))
        indices = np.arange(X.shape[0])

        start_time = time()
        for i in range(self.epoch):
            np.random.shuffle(indices)
            for j in range(step):
                self.optimizer.zero_grad()
                output = self.model.forward(X_[indices[j * self.batch: (j + 1) * self.batch]])
                loss = self.criterion(output, y_true[indices[j * self.batch: (j + 1) * self.batch]])
                loss.backward()
                self.optimizer.step()
            y_pred = self.model.forward(X_)

            self.writer.add_scalar('loss', self.criterion(y_pred, y_true).item(), i)
            for k, metric in self.metrics.items():
                target_metric = metric(y_pred, y_true)
                if target_metric == 0:
                    print(y_pred, y_true)
                self.writer.add_scalar(k, metric(y_pred, y_true), i)
        print(f'High level model trained in {time() - start_time}')
        return self.model
