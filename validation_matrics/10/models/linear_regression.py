import math
from .general_model import GeneralModel
import torch
from .model_utils import config


class LinearRegression(GeneralModel):
    def __init__(
        self,
        batch_size=config['BATCH_SIZE'],
        learning_rate=config['LEARNING_RATE'],
        reg_type=config['REG_TYPE'],
        c=config['C'],
        use_gpu=config['USE_GPU']
    ):
        GeneralModel.__init__(self, use_gpu)
        self.coef = None
        self.loss = math.inf
        self.batch_size = batch_size
        self.epoch_num = 1
        self.mean = None
        self.std = None
        self.learning_rate = learning_rate
        self.reg_type = reg_type
        self.c = c

    def _predict(self, X):
        return torch.matmul(X, self.coef)

    def fit(self, X, y, epochs=100):
        N, k_num = X.shape
        N_y,  = y.shape

        assert N == N_y

        y_train = torch.tensor(y.values, dtype=torch.float32, device=self.device)
        y_train.share_memory_()
        X_train = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        X_train.share_memory_()
        normalized_X, self.std, self.mean = self._normalize(X_train)
        x_train = self._add_intercept(normalized_X)
        self.coef = self._init_coef(k_num)
        self.batch_size = min(self.batch_size, N)

        while self.loss > 0.0001 and self.epoch_num < epochs:
            self._train(x_train, y_train)

        return self

    def _train(self, x_train, y_train):
        for X_train_batch, y_train_batch in self._get_minibatch(x_train, y_train, self.batch_size):
            y_pred = self._predict(X_train_batch)
            self.loss = self._get_loss(y_pred, y_train_batch)
            self.coef -= self.learning_rate * self._get_gradients(X_train_batch, y_train_batch, y_pred)

        self.epoch_num += 1

    def _get_loss(self, y, y_pred):
        penalty = self._get_penalty()

        return (1 / (2 * y.shape[0])) * torch.sqrt(torch.sum(torch.pow(y_pred - y, 2))) + penalty

    def _get_gradients(self, X, y, y_pred):
        return (1 / X.shape[0]) * (torch.matmul(torch.transpose(X, 0, 1), y_pred - y))

    def predict(self, X):
        N, k_num = X.shape
        assert k_num == self.coef.shape[0] - 1

        normalized_X = (torch.tensor(X.values, dtype=torch.float32).to(device=self.device) - self.mean) / self.std

        return self._predict(self._add_intercept(normalized_X))
