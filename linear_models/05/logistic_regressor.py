from sklearn.base import ClassifierMixin, BaseEstimator
from utils import sigmoid
import numpy as np


class MyLogisticRegression(ClassifierMixin, BaseEstimator):

    """
    Logistic regressor implementation
    """

    def __init__(
        self,
        learning_rate,
        num_iterations,
        C,
        verbose=True,
        fit_intercept=True,
        print_steps=1000,
    ):

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.C = C
        self.verbose = verbose
        self.weights = None
        self.print_steps = print_steps

    def validate_inputs(self, X, Y):

        assert len(X.shape) == 2
        assert len(X) == len(Y)
        return X

    def initialize_weights(self, input_shape):
        self.weights = np.random.normal(size=input_shape)[..., None]

    def binary_crossentropy(self, preds, Y):
        return (-Y * np.log(preds) - (1 - Y) * np.log(1 - preds)).mean()

    def fit(self, X, Y):
        inputs = self.validate_inputs(X, Y)

        if self.fit_intercept:
            inputs = np.concatenate((np.ones((len(X), 1)), X), axis=1)

        self.initialize_weights(inputs.shape[1])

        for i in range(self.num_iterations):

            logits = np.dot(inputs, self.weights)

            probs = sigmoid(logits)

            gradients = (
                np.dot(inputs.T, (probs - Y)) / len(X)
                + (1 / (self.C * len(X))) * self.weights
            )

            self.weights -= self.learning_rate * gradients

            if self.verbose and i % self.print_steps == 0:

                preds = self.predict_proba(X)
                acc = self.score(X, Y)

                loss = self.binary_crossentropy(preds, Y)

                print(f"Log_loss at {i} step is {loss} \t train_accuracy is {acc}")

    def predict_proba(self, X):

        inputs = X
        if self.fit_intercept:
            inputs = np.concatenate((np.ones((len(X), 1)), X), axis=1)

        return sigmoid(np.dot(inputs, self.weights))

    def predict(self, X, binarization_threshold=0.5):
        return (self.predict_proba(X) > binarization_threshold).astype(np.uint8)
