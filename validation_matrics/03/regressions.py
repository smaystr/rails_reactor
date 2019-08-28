import numpy
from utils import sigmoid
from utils import benchmark_prediction


class LogisticRegression:
    def __init__(self,
                 X_train=None,
                 y_train=None,
                 alpha=None,
                 regularisation=False,
                 lmd=None,
                 max_iter=10000
                 ):
        """
        Logistic regression constructor
        :type X_train: numpy.ndarray
        :type y_train: numpy.ndarray
        :type alpha: float
        :type regularisation: bool
        :type lmd: float
        :type max_iter: int
        """
        self.__X_train = X_train
        self.__y_train = y_train
        if (not (self.__X_train is None)) and (not (self.__y_train is None)):
            if self.__X_train.shape[0] != self.__y_train.shape[1]:
                raise AttributeError('Fatal error. The X_train and y_train shapes are different. (problem in axis=0)')
            self.__theta = numpy.ones((self.__X_train.shape[0], self.__X_train.shape[1]))
        self.max_iter = int(max_iter)
        self.__regularisation = regularisation
        if alpha is None:
            raise AttributeError('Specify the alpha hyperparameter before training your model')
        self.alpha = alpha
        if self.__regularisation:
            if lmd is None:
                raise AttributeError(
                    'Specify the lambda hyperparameter before training your model. You\'ve chosen the regularisation modification of the algorithm.')
            self.lmd = lmd

    def fit(self, X_train, y_train):
        self.__X_train = X_train
        self.__y_train = y_train
        self.__theta = numpy.ones((self.__X_train.shape[1] + 1, 1))

    def __hypothesis(self):
        return sigmoid(
            numpy.dot(self.__theta.T, numpy.c_[numpy.ones((len(self.__X_train), 1)), self.__X_train].T)
        )

    def __cost_function(self):
        hypothesis = self.__hypothesis().T
        return (-1 / (len(self.__X_train))) * (numpy.dot(self.__y_train.T, numpy.log(hypothesis)) + numpy.dot((1 - self.__y_train.T), numpy.log(1 - hypothesis)))

    def __cost_function_with_regularisation(self):
        hypothesis = self.__hypothesis().T
        return (-1 / (len(self.__X_train))) * (
                numpy.dot(self.__y_train.T, numpy.log(hypothesis)) + numpy.dot((1 - self.__y_train.T), numpy.log(
            1 - hypothesis))) + self.lmd * numpy.power(self.__theta, 2).sum()

    def __cost_function_derivative(self):
        return (1 / len(self.__X_train)) * (
            numpy.dot((numpy.c_[numpy.ones((len(self.__X_train), 1)), self.__X_train]).T,
                      self.__hypothesis().T - self.__y_train))

    def __gradient_descend(self) -> tuple:
        cost_function_history = []
        for iteration in range(self.max_iter):
            self.__theta = self.__theta - (self.alpha / len(self.__X_train)) * self.__cost_function_derivative()
            cost_function_history.append(self.__cost_function())
        return cost_function_history, self.__theta

    def __gradient_descend_with_regularisation(self) -> tuple:
        cost_function_history = []
        for iteration in range(self.max_iter):
            self.__theta = self.__theta * (1 - (self.alpha * (
                    self.lmd / len(self.__X_train)))) - self.alpha * self.__cost_function_derivative()
            cost_function_history.append(self.__cost_function_with_regularisation())
        return cost_function_history, self.__theta

    def __predict_probs(self, X_test):
        return sigmoid(numpy.dot(X_test, self.__theta))

    @benchmark_prediction()
    def predict(self, X_test, threshold=.5):
        if self.__X_train is None or self.__y_train is None:
            raise AttributeError('Fit the training dataset to the model before training it!')
        if self.__regularisation:
            loss, self.__theta = self.__gradient_descend_with_regularisation()
        else:
            loss, self.__theta = self.__gradient_descend()
        return loss, self.__predict_probs(numpy.c_[numpy.ones((len(X_test), 1)), X_test]) >= threshold

    def get_weights(self):
        return self.__theta


class LinearRegression:
    def __init__(self,
                 X_train=None,
                 y_train=None,
                 alpha=None,
                 regularisation=False,
                 lmd=None,
                 max_iter=10000
                 ):
        """
        Linear regression constructor
        :type X_train: numpy.ndarray
        :type y_train: numpy.ndarray
        :type alpha: float
        :type regularisation: bool
        :type lmd: float
        :type max_iter: int
        """
        self.__X_train = X_train
        self.__y_train = y_train
        if (not (self.__X_train is None)) and (not (self.__y_train is None)):
            if self.__X_train.shape[0] != self.__y_train.shape[1]:
                raise AttributeError('Fatal error. The X_train and y_train shapes are different. (problem in axis=0)')
            self.__theta = numpy.ones((self.__X_train.shape[0], self.__X_train.shape[1]))
        self.__regularisation = regularisation
        self.__max_iter = int(max_iter)
        if alpha is None:
            raise AttributeError('Specify the alpha hyperparameter before training your model.')
        self.alpha = alpha
        if self.__regularisation:
            if lmd is None:
                raise AttributeError(
                    'Specify the lambda hyperparameter before training your model. You\'ve chosen the regularisation modification of the algorithm.')
            self.lmd = lmd

    def fit(self, X_train, y_train):
        self.__X_train = X_train
        self.__y_train = y_train
        self.__theta = numpy.random.rand(len(self.__X_train[0]) + 1, 1)

    def __hypothesis(self):
        return numpy.dot(self.__theta.T, numpy.c_[numpy.ones((len(self.__X_train), 1)), self.__X_train].T)

    def __cost_function(self):
        return (1 / (2 * len(self.__X_train))) * numpy.power((self.__hypothesis().T - self.__y_train), 2)

    def __cost_function_with_regularisation(self):
        regularisation = 0.
        for index in range(len(self.__theta)):
            regularisation += numpy.power(self.__theta[index], 2)
        regularisation *= self.lmd
        return (1 / (2 * len(self.__X_train))) * (
                numpy.power((self.__hypothesis().T - self.__y_train), 2) + regularisation)

    def __cost_function_derivative(self):
        return (1 / len(self.__X_train)) * (
            numpy.dot((numpy.c_[numpy.ones((len(self.__X_train), 1)), self.__X_train]).T,
                         self.__hypothesis().T - self.__y_train))

    def __gradient_descend(self) -> tuple:
        cost_function_history = []
        for iteration in range(self.__max_iter):
            self.__theta = self.__theta - (self.alpha / len(self.__X_train)) * self.__cost_function_derivative()
            cost_function_history.append(self.__cost_function())
        return cost_function_history, self.__theta

    def __gradient_descend_with_regularisation(self) -> tuple:
        cost_function_history = []
        for iteration in range(self.__max_iter):
            self.__theta = self.__theta * (1 - self.alpha * (self.lmd / len(self.__X_train))) - (
                    self.alpha / len(self.__X_train) * self.__cost_function_derivative())
        cost_function_history.append(self.__cost_function_with_regularisation())
        return cost_function_history, self.__theta

    @benchmark_prediction()
    def predict(self, X_test):
        if self.__X_train is None or self.__y_train is None:
            raise AttributeError('Fit the training dataset to the model before training it!')
        if self.__regularisation:
            loss, self.__theta = self.__gradient_descend_with_regularisation()
        else:
            loss, self.__theta = self.__gradient_descend()
        return loss, self.__predict_probs(numpy.c_[numpy.ones((len(X_test), 1)), X_test]).T

    def __predict_probs(self, X_test):
        return numpy.dot(X_test, self.__theta)

    def get_X_y(self):
        return self.__X_train, self.__y_train

    def get_weights(self):
        return self.__theta
