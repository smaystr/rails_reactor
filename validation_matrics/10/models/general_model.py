import math
import torch
from .model_utils import config


class GeneralModel:

    reg_type = config['REG_TYPE']
    coef = []
    c = config['C']
    device = torch.device('cpu')
    epochs_max_num = config['EPOCHS_MAX_NUM']

    def __init__(self, use_gpu):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')

    @staticmethod
    def _get_minibatch(X, y, batch_size):
        N = X.shape[0]
        batches_qty = math.ceil(N / batch_size)

        for batch_num in range(batches_qty):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, N)

            X_batch = X[start: end]
            y_batch = y[start: end]

            yield X_batch, y_batch

    @staticmethod
    def _normalize(array):
        std = torch.std(array)
        mean = torch.mean(array)
        normalized = (array - mean) /std

        return normalized, std, mean

    def _add_intercept(self, X):
        ones = torch.ones((X.shape[0], 1), dtype=X.dtype, device=self.device)
        return torch.cat([ones, X], dim=1).to(self.device)

    def _init_coef(self, k_num):
        coef = torch.rand(k_num + 1, dtype=torch.float32) * torch.sqrt(torch.tensor(1 / (k_num + 2)))
        coef = coef.to(device=self.device)
        return coef

    def _get_penalty(self):
        if self.reg_type is None:
            return 0

        if self.reg_type == 'l1':
            return torch.mean(torch.abs(self.coef[1:])) / self.c

        if self.reg_type == 'l2':
            return torch.mean(torch.rsqrt(self.coef[1:])) / self.c



    def fit(self, X, y, epochs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
