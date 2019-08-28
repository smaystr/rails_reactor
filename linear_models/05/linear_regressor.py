from sklearn.base import RegressorMixin, BaseEstimator
import numpy as np


class MyLinearRegression(RegressorMixin, BaseEstimator):

    """
    Linear regressor implementation with L2 regularization (aka RidgeRegression)

    """

    def __init__(
        self,
        learning_rate,
        num_iterations,
        lam,
        verbose=True,
        fit_intercept=True,
        print_steps=1000,
    ):

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.lam = lam
        self.verbose = verbose
        self.weights = None
        self.print_steps = print_steps

    def validate_inputs(self, X, Y):

        assert len(X.shape) == 2
        assert len(X) == len(Y)
        return X

    def initialize_weights(self, input_shape):
        self.weights = np.random.normal(size=input_shape)[..., None]

    def mse(self, preds, Y):
        return (np.square(Y - preds)).mean()

    def fit(self, X, Y):
        inputs = self.validate_inputs(X, Y)

        if self.fit_intercept:
            inputs = np.concatenate((np.ones((len(X), 1)), X), axis=1)

        self.initialize_weights(inputs.shape[1])

        for i in range(self.num_iterations):

            logits = np.dot(inputs, self.weights)

            gradients = (
                np.dot(inputs.T, (logits - Y)) / len(X)
                + (self.lam / len(X)) * self.weights
            )

            self.weights -= self.learning_rate * gradients

            if self.verbose and i % self.print_steps == 0:

                preds = self.predict(X)
                loss = self.mse(preds, Y)

                print(f"MSE at {i} step is {loss}\t RMSE is {np.sqrt(loss)}")

    def predict(self, X):

        inputs = X
        if self.fit_intercept:
            inputs = np.concatenate((np.ones((len(X), 1)), X), axis=1)

        return np.dot(inputs, self.weights)
