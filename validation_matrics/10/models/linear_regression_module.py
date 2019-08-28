import math
import torch
import torch.nn as nn
from .general_model import GeneralModel
from .model_utils import config


class LinearRegressionModule(nn.Module):

    def __init__(self, features_num):
        super(LinearRegressionModule, self).__init__()
        self.fc = nn.Linear(features_num, 1)

    def forward(self, X):
        return self.fc(X)


class LinearRegression(GeneralModel):
    def __init__(
        self,
        batch_size=config['BATCH_SIZE'],
        learning_rate=config['LEARNING_RATE'],
        use_gpu=config['USE_GPU'],
    ):
        GeneralModel.__init__(self, use_gpu)
        self.coef = None
        self.loss = math.inf
        self.batch_size = batch_size
        self.epoch_num = 1
        self.mean = None
        self.std = None
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None

    def fit(self, X, y):
        N, k_num = X.shape
        N_y,  = y.shape

        assert N == N_y

        X_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        X_tensor.share_memory_()
        x_train, self.std, self.mean = self._normalize(X_tensor)

        y_train = torch.tensor(y.values, dtype=torch.float32, device=self.device)
        y_train.share_memory_()

        self.coef = self._init_coef(k_num)
        self.batch_size = min(self.batch_size, N)

        self.model = LinearRegressionModule(features_num=k_num)
        self.model.to(device=self.device)

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)
        self.optimizer.zero_grad()

        while self.loss > 0.000001 and self.epoch_num < self.epochs_max_num:
            for X_train_batch, y_train_batch in self._get_minibatch(x_train, y_train, self.batch_size):
                y_pred = self.model(X_train_batch)
                loss = nn.MSELoss()(y_pred.reshape(y_pred.shape[0]), y_train_batch)
                loss.backward()
                self.optimizer.step()
            self.loss = float(loss)
            self.epoch_num += 1

        self.coef = self.model.parameters()

        return self

    def predict(self, X):
        normalized_X = (torch.tensor(X.values, dtype=torch.float32).to(device=self.device) - self.mean) / self.std
        return self.model(normalized_X).reshape(X.shape[0])
