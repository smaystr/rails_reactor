import numpy as np
import math


class GeneralModel():
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
        std = np.std(array, axis=0)
        mean = np.mean(array, axis=0)
        normalized = (array - mean) /std

        return normalized, std, mean

    @staticmethod
    def _add_intercept(X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    @staticmethod
    def _init_coef(k_num):
        return np.random.rand(k_num + 1, ) * np.sqrt(1 / (k_num + 2))

    def fit(self, X, y, epochs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
