import math
from .general_model import GeneralModel
import torch
from .model_utils import config


class LogisticRegression(GeneralModel):
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

    def fit(self, X, y, epochs=100):
        N, k_num = X.shape
        N_y,  = y.shape

        assert N == N_y

        normalized_X, self.std, self.mean = self._normalize(torch.tensor(X.values, dtype=torch.float32, device=self.device).share_memory_())
        x_train = self._add_intercept(normalized_X)
        self.coef = self._init_coef(k_num)
        self.batch_size = min(self.batch_size, N)

        while self.loss > 0.000001 and self.epoch_num < epochs:
            self._train(x_train, torch.tensor(y.values, dtype=torch.float32, device=self.device).share_memory_())

        return self

    def _train(self, x_train, y_train):
        for X_train_batch, y_train_batch in self._get_minibatch(x_train, y_train, self.batch_size):
            y_pred = self._proba(X_train_batch)
            self.loss = float(self._get_loss(y_train_batch, y_pred))
            self.coef -= self.learning_rate * self._get_gradients(X_train_batch, y_train_batch, y_pred)

        self.epoch_num += 1

    def _get_loss(self, y, y_pred):
        # https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11
        # cost = -y * np.log(y_pred) - (1-y) * np.log(1 - y_pred)
        m = y.shape[0]
        penalty = self._get_penalty()
        t1 = -y * torch.log(y_pred)
        return (1 / m) * torch.sum(-y * torch.log(y_pred) - (1 - y) * torch.log(1 - y_pred)) + penalty

    def _get_gradients(self, X, y, y_pred):
        diff = 1 / (1 + torch.exp(y_pred)) - y

        return (1 / X.shape[0]) * (torch.matmul(torch.transpose(X, 0, 1), diff))

    def _predict(self, X):
        return torch.round(self._proba(X))

    def _proba(self, X):
        N, k_num = X.shape
        assert k_num == self.coef.shape[0]
        return self._sigmoid(torch.matmul(X, self.coef))

    def predict(self, X):
        return torch.round(self.proba(X))

    def proba(self, X):
        N, k_num = X.shape
        assert k_num == self.coef.shape[0] - 1

        normalized_X = (torch.tensor(X.values, dtype=torch.float32).to(device=self.device) - self.mean) / self.std

        return self._sigmoid(torch.matmul(self._add_intercept(normalized_X), self.coef))

    def _sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
