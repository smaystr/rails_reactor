import torch
from torch import sigmoid
from utils import accuracy, timeit, load_config

config = load_config()


class TorchLinearRegression:
    def __init__(
        self,
        learning_rate,
        num_iterations,
        lam,
        verbose=True,
        fit_intercept=True,
        print_steps=100,
        use_gpu=config.get("USE_GPU"),
        batch_size=config.get("BATCH_SIZE"),
    ):

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.lam = lam
        self.verbose = verbose
        self.weights = None
        self.print_steps = print_steps
        self.use_gpu = use_gpu
        self.batch_size = batch_size

    def validate_inputs(self, features, labels):

        assert len(features.shape) == 2
        assert len(features) == len(labels)

        if self.use_gpu:
            features = features.cuda()
            labels = labels.cuda()

        return features, labels

    def initialize_weights(self, input_shape):
        self.weights = torch.randn(input_shape)[..., None]
        if self.use_gpu:
            self.weights = self.weights.cuda()

    def mse(self, Y_pred, Y_true):
        return torch.mean(torch.pow(Y_pred - Y_true, 2))

    def __gen_batches(self, features, labels, indexes):
        return (
            [features[index] for index in indexes],
            [labels[index] for index in indexes],
        )

    @timeit
    def fit(self, features, labels):
        features, labels = self.validate_inputs(features, labels)

        if self.fit_intercept:
            ones = torch.ones((len(features), 1))

            if self.use_gpu:
                ones = ones.cuda()

            inputs = torch.cat((ones, features), dim=1)
        else:
            inputs = features

        self.initialize_weights(inputs.shape[1])

        indexes = torch.arange(len(inputs))

        batch_indexes = torch.split(indexes, self.batch_size)
        X_batches, Y_batches = self.__gen_batches(inputs, labels, batch_indexes)

        for i in range(self.num_iterations):

            for features_batch, Y_batch in zip(X_batches, Y_batches):

                logits = torch.matmul(features_batch, self.weights)

                gradients = (
                    torch.matmul(features_batch.t(), (logits - Y_batch))
                    / len(features_batch)
                    + (self.lam / len(features_batch)) * self.weights
                )

                self.weights -= self.learning_rate * gradients

            if self.verbose and i % self.print_steps == 0:

                preds = self.predict(features)
                loss = self.mse(preds, labels)

                print(f"MSE at {i} step is {loss:.1f}\t RMSE is {torch.sqrt(loss):.4f}")

    def predict(self, features):
        inputs = features
        if self.fit_intercept:
            ones = torch.ones((len(features), 1))
            if self.use_gpu:
                ones = ones.cuda()

            inputs = torch.cat((ones, features), dim=1)

        return torch.matmul(inputs, self.weights)


class TorchLogisticRegression:
    def __init__(
        self,
        learning_rate,
        num_iterations,
        C,
        batch_size=config.get("BATCH_SIZE"),
        verbose=True,
        fit_intercept=True,
        print_steps=100,
        use_gpu=config.get("USE_GPU"),
    ):

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.C = C
        self.verbose = verbose
        self.weights = None
        self.print_steps = print_steps
        self.use_gpu = use_gpu
        self.batch_size = batch_size

    def validate_inputs(self, features, labels):

        assert len(features.shape) == 2
        assert len(features) == len(labels)

        if self.use_gpu:
            features = features.cuda()
            labels = labels.cuda()

        return features, labels

    def initialize_weights(self, input_shape):
        self.weights = torch.randn(input_shape)[..., None]
        if self.use_gpu:
            self.weights = self.weights.cuda()

    def binary_cross_entropy(self, Y_pred, Y_true):
        return (
            -Y_true * torch.log(Y_pred) - (1 - Y_true) * torch.log(1 - Y_pred)
        ).mean()

    def __gen_batches(self, features, labels, indexes):
        return (
            [features[index] for index in indexes],
            [labels[index] for index in indexes],
        )

    @timeit
    def fit(self, features, labels):
        features, labels = self.validate_inputs(features, labels)

        if self.fit_intercept:
            ones = torch.ones((len(features), 1))

            if self.use_gpu:
                ones = ones.cuda()

            inputs = torch.cat((ones, features), dim=1)
        else:
            inputs = features

        self.initialize_weights(inputs.shape[1])

        indexes = torch.arange(len(inputs))

        batch_indexes = torch.split(indexes, self.batch_size)
        X_batches, Y_batches = self.__gen_batches(inputs, labels, batch_indexes)

        for i in range(self.num_iterations):

            for X_batch, Y_batch in zip(X_batches, Y_batches):

                logits = torch.matmul(X_batch, self.weights)

                preds = sigmoid(logits)

                gradients = (
                    torch.matmul(X_batch.t(), (preds - Y_batch)) / len(X_batch)
                    + (1 / (self.C * len(features))) * self.weights
                )

                self.weights -= self.learning_rate * gradients

            if self.verbose and i % self.print_steps == 0:

                preds = self.predict_proba(features)
                loss = self.binary_cross_entropy(preds, labels)
                acc = accuracy(preds, labels)

                print(f"Log_loss at {i} step is {loss:.4f}\t Acc is {acc:.4f}")

    def predict_proba(self, features):
        inputs = features
        if self.fit_intercept:
            ones = torch.ones((len(features), 1))
            if self.use_gpu:
                ones = ones.cuda()

            inputs = torch.cat((ones, features), dim=1)

        return sigmoid(torch.matmul(inputs, self.weights))

    def predict(self, X, binarization_threshold=0.5):
        return torch.as_tensor(
            self.predict_proba(X) > binarization_threshold, dtype=torch.uint8
        )
