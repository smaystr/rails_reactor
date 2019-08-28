import torch
from tqdm import tqdm
from math import ceil


class LinearRegression:
    def __init__(
        self,
        lr=0.0001,
        tol=1e-3,
        fit_intercept=True,
        verbose_rounds=200,
        verbose=False,
        penalty="l2",
        C=1,
        max_iter=2000,
        batch_size=10,
        cuda=False,
    ):
        self.lr = lr
        self.C = C
        self.fit_intercept = fit_intercept
        self.not_training = True
        self.tol = tol
        self.max_iter = max_iter
        self.verbose_rounds = verbose_rounds
        self.verbose = verbose
        self.batch_size = batch_size
        self.cuda = cuda
        self.loss_history = []

        penalties = set(["l1", "l2"])
        penalty = penalty.lower()

        if penalty in penalties:
            self.penalty = penalty
        else:
            raise Exception(
                f"Penalty should be {penalties}. The the penalty was: {penalty}"
            )

    def add_intercept(self, X):
        if self.fit_intercept:
            X = torch.cat((torch.ones(X.shape[0], 1), X), dim=1)
        return X

    def init_weights(self, cols):
        self.weights = self.to_cuda(torch.rand(cols, 1))

    def cost(self, X, y):
        prediction_error = torch.matmul(X, self.weights) - y
        return torch.sum((prediction_error * prediction_error) / (2 * X.shape[0])) + self.regularize()

    def to_cuda(self, tensor):
        if self.cuda:
            return tensor.cuda()
        return tensor

    def regularize_step(self):
        if self.penalty == "l1":
            return self.to_cuda(torch.sign(a)) / self.C
        return (2 / self.C) * self.weights

    def regularize(self):
        if self.penalty == "l1":
            return torch.sum((1 / self.C) * torch.abs(self.weights))
        return torch.sum((1 / self.C) * (self.weights * self.weights))

    def step(self, X_batch, y_batch):
        self.weights -= (
            self.lr
            / X_batch.shape[0]
            * torch.matmul(
                torch.t(X_batch), torch.matmul(X_batch, self.weights) - y_batch
            )
            + self.regularize_step()
        )

    def fit(self, X, y):
        self.size = len(X)

        X = self.to_cuda(self.add_intercept(X))
        y = self.to_cuda(y.reshape(len(y), 1).type(torch.FloatTensor))

        self.not_training = False
        self.init_weights(X.shape[1])

        self.batches_in_epoch = int(ceil(self.size / self.batch_size))

        indices = self.to_cuda(torch.arange(self.size))
        for i in range(self.max_iter):
            epoch_start_error = self.cost(X, y)

            self.loss_history.append(torch.sum(epoch_start_error).item())

            for batch in tqdm(
                torch.split(indices, self.batches_in_epoch), disable=not self.verbose
            ):

                self.step(
                    self.to_cuda(torch.index_select(X, 0, batch)), self.to_cuda(torch.index_select(y, 0, batch))
                )

            epoch_end_error = self.cost(X, y)
            trained_difference = torch.abs(torch.sum(epoch_start_error - epoch_end_error))

            if self.verbose and i % self.verbose_rounds == 0:
                print(f"Error at round {i}: {torch.sum(epoch_end_error)}")
            if trained_difference < self.tol:
                break
        self.not_training = True

    def evaluate(self, X, y):
        if self.not_training:
            X = self.to_cuda(self.add_intercept(X))
        return torch.sum((torch.pow(torch.matmul(X, self.weights) - y), 2) / (len(X)))

    def predict(self, X):
        if self.not_training:
            X = self.to_cuda(self.add_intercept(X))
        return torch.matmul(X, self.weights)


class LogisticRegression:
    def __init__(
        self,
        lr=0.001,
        C=1,
        max_iter=250,
        fit_intercept=True,
        penalty="l2",
        eps=1e-12,
        verbose_rounds=200,
        verbose=False,
        tol=1e-5,
        batch_size=20,
        cuda=False,
    ):
        self.lr = lr
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.eps = eps
        self.verbose_rounds = verbose_rounds
        self.verbose = verbose
        self.not_training = True
        self.tol = tol
        self.batch_size = batch_size
        self.cuda = cuda
        self.loss_history = []

        penalties = set(["l1", "l2"])
        penalty = penalty.lower()

        if penalty in penalties:
            self.penalty = penalty
        else:
            raise Exception(
                f"Penalty should be {penalties}. The the penalty was: {penalty}"
            )

    def init_weights(self, cols):
        self.weights = self.to_cuda(torch.rand(cols, 1))

    def loss(self, X, y):
        sigm = self.predict_proba(X)
        return torch.mean(-y * torch.log(sigm) - (1 - y) * torch.log(1 - sigm)) + self.regularize()

    def add_intercept(self, X):
        if self.fit_intercept:
            X = torch.cat((torch.ones(X.shape[0], 1), X), dim=1)
        return X

    def to_cuda(self, tensor):
        if self.cuda:
            return tensor.cuda()
        return tensor

    def regularize_step(self):
        if self.penalty == "l1":
            return torch.sign(a) / (self.C)
        return (2 / self.C) * self.weights

    def regularize(self):
        if self.penalty == "l1":
            return torch.sum((1 / self.C) * torch.abs(self.weights))
        return torch.sum((1 / self.C) * (self.weights * self.weights))

    def step(self, X_batch, y_batch):
        objective = self.predict_proba(X_batch)

        self.weights -= self.lr * (
            (1 / len(X_batch)) * torch.matmul(torch.t(X_batch), objective - y_batch)
            + self.regularize_step()
        )

    def fit(self, X, y):
        self.size = len(X)

        X = self.to_cuda(self.add_intercept(X))
        y = self.to_cuda(y.reshape(len(y), 1).type(torch.FloatTensor))

        self.not_training = False

        self.init_weights(X.shape[1])

        self.batches_in_epoch = int(ceil(self.size / self.batch_size))

        indices = self.to_cuda(torch.arange(self.size))

        for i in range(self.max_iter):
            epoch_start_error = self.loss(X, y)

            self.loss_history.append(torch.sum(epoch_start_error).item())

            for batch in tqdm(
                torch.split(indices, self.batches_in_epoch), disable=not self.verbose
            ):
                self.step(
                    self.to_cuda(torch.index_select(X, 0, batch)), self.to_cuda(torch.index_select(y, 0, batch))
                )

            epoch_end_error = self.loss(X, y)
            trained_difference = torch.abs(
                torch.sum(epoch_start_error - epoch_end_error)
            )
            if i % self.verbose_rounds == 0:
                print(f"Loss after {i} steps: {epoch_end_error}")

            if trained_difference < self.tol:
                break

        self.not_training = True

    def predict_proba(self, X):
        if self.not_training:
            X = self.to_cuda(self.add_intercept(X))

        return 1 / (1 + torch.exp(-1 * torch.matmul(X, self.weights) + self.eps))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold) * 1.0
