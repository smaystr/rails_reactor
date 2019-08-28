import torch


class LogisticRegression:
    def __init__(self,
                 X_train=None,
                 y_train=None,
                 alpha=None,
                 regularisation=False,
                 lmd=None,
                 max_iter=10000,
                 use_SGD=False,
                 batch_size=None,
                 use_gpu=False
                 ):
        """
        Logistic regression constructor
        :type X_train: torch.tensor
        :type y_train: torch.tensor
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
            self.__theta = torch.ones(self.__X_train.shape[0], self.__X_train.shape[1], dtype=torch.float64)
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
        self.__use_gpu = use_gpu
        self.__use_SGD = use_SGD
        self.__batch_size = batch_size
        if self.__use_SGD and self.__batch_size is None:
            raise AttributeError(
                'Batch size is not defined, use_SGD mode is on. Define the size firstly before training the model.')


def fit(self, X_train, y_train):
    self.__X_train = X_train
    self.__y_train = y_train
    self.__theta = torch.rand((self.__X_train.shape[1] + 1, 1), dtype=torch.float64)
    if self.__use_gpu:
        self.__theta.cuda()
        self.__X_train.cuda()
        self.__y_train.cuda()


def __hypothesis(self):
    return torch.sigmoid(
        torch.mm(self.__theta.t(),
                 torch.cat((torch.ones(len(self.__X_train), 1, dtype=torch.float64), self.__X_train), dim=1).t())
    )


def __cost_function(self):
    hypothesis = self.__hypothesis().t()
    if self.__regularisation:
        if self.__use_SGD:
            return (1 / 2) * (torch.mm(self.__y_train.t(), torch.log(hypothesis)) + torch.mm((1 - self.__y_train.t()),
                                                                                             torch.log(
                                                                                                 1 - hypothesis))) + self.lmd * torch.pow(
                self.__theta, 2).sum()
        else:
            return (-1 / (len(self.__X_train))) * (
                    torch.mm(self.__y_train.t(), torch.log(hypothesis)) + torch.mm((1 - self.__y_train.t()), torch.log(
                1 - hypothesis))) + self.lmd * torch.pow(self.__theta, 2).sum()
    else:
        if self.__use_SGD:
            return (1 / 2) * torch.pow((self.__hypothesis() - self.__y_train), 2)
        else:
            return (-1 / (len(self.__X_train))) * (
                    torch.mm(self.__y_train.t(), torch.log(hypothesis)) + torch.mm((1 - self.__y_train.t()),
                                                                                   torch.log(1 - hypothesis)))


def __cost_function_derivative(self):
    if self.__use_SGD:
        return torch.mm(torch.cat((torch.ones(len(self.__X_train), 1, dtype=torch.float64), self.__X_train), 1).t(),
                        self.__hypothesis().t() - self.__y_train)
    else:
        return (1 / len(self.__X_train)) * (
            torch.mm(torch.cat((torch.ones(len(self.__X_train), 1, dtype=torch.float64), self.__X_train), 1).t(),
                     self.__hypothesis().t() - self.__y_train))


def __gen_batches(self, X, y, indices):
    return (
        [X[batch_indices] for batch_indices in indices],
        [y[batch_indices] for batch_indices in indices]
    )


def __gradient_descend(self) -> tuple:
    cost_function_history = []
    for iteration in range(self.max_iter):
        if self.__use_SGD:
            self.__theta = self.__theta * (1 - (self.alpha * (
                    self.lmd / len(self.__X_train)))) - self.alpha * self.__cost_function_derivative()
        else:
            self.__theta = self.__theta - (self.alpha / len(self.__X_train)) * self.__cost_function_derivative()
        cost_function_history.append(self.__cost_function())
    return cost_function_history, self.__theta


def __predict_probs(self, X_test):
    return torch.sigmoid(torch.mm(X_test, self.__theta))


def predict(self, X_test, threshold=.5):
    if self.__X_train is None or self.__y_train is None:
        raise AttributeError('Fit the training dataset to the model before testing it!')
    loss, self.__theta = self.__gradient_descend()
    return loss, self.__predict_probs(
        torch.cat((torch.ones((len(X_test), 1), dtype=torch.float64), X_test), 1)) >= threshold


class LinearRegression:
    def __init__(self,
                 X_train=None,
                 y_train=None,
                 alpha=None,
                 regularisation=False,
                 lmd=None,
                 max_iter=10000,
                 use_SGD=None,
                 batch_size=None,
                 use_gpu=False
                 ):
        """
        Linear regression constructor
        :type X_train: torch.tensor
        :type y_train: torch.tensor
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
            self.__theta = torch.rand(self.__X_train.shape[0], self.__X_train.shape[1], dtype=torch.float64)
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
        self.__use_gpu = use_gpu
        self.__use_SGD = use_SGD
        self.__batch_size = batch_size
        if self.__use_SGD and self.__batch_size is None:
            raise AttributeError(
                'Batch size is not defined, use_SGD mode is on. Define the size firstly before training the model.')

    def fit(self, X_train, y_train):
        self.__X_train = X_train
        self.__y_train = y_train
        self.__theta = torch.rand((len(self.__X_train[0]) + 1, 1), dtype=torch.float64)

    def __hypothesis(self):
        return torch.mm(self.__theta.t(),
                        torch.cat((torch.ones((len(self.__X_train), 1), dtype=torch.float64), self.__X_train), 1).t())

    def __cost_function(self):
        return (1 / (2 * len(self.__X_train))) * torch.pow((self.__hypothesis().t() - self.__y_train), 2)

    def __stochatic_cost_function(self):
        return (1 / 2) * torch.pow((self.__hypothesis() - self.__y_train), 2)

    def __cost_function_with_regularisation(self):
        regularisation = 0.
        for index in range(len(self.__theta)):
            regularisation += torch.pow(self.__theta[index], 2)
        regularisation *= self.lmd
        return (1 / (2 * len(self.__X_train))) * (
                torch.pow((self.__hypothesis().t() - self.__y_train), 2) + regularisation)

    def __cost_function_derivative(self):
        return (1 / len(self.__X_train)) * (
            torch.mm(torch.cat((torch.ones(len(self.__X_train), 1, dtype=torch.float64), self.__X_train), 1).t(),
                     self.__hypothesis().t() - self.__y_train))

    def __stochatic_cost_function_derivative(self):
        return (torch.mm(torch.cat((torch.ones(len(self.__X_train), 1, dtype=torch.float64), self.__X_train), 1).t(),
                         self.__hypothesis().t() - self.__y_train))

    def __gradient_descend(self) -> tuple:
        cost_function_history = []
        for iteration in range(self.__max_iter):
            self.__theta = self.__theta - (self.alpha / len(self.__X_train)) * self.__cost_function_derivative()
            cost_function_history.append(self.__cost_function())
        return cost_function_history, self.__theta

    def __stochatic_gradient_descend(self) -> tuple:
        cost_function_history = []
        for iteration in range(self.__max_iter):
            self.__theta = self.__theta - self.alpha * self.__stochatic_cost_function_derivative()
            cost_function_history.append(self.__cost_function())
        return cost_function_history, self.__theta

    def __gradient_descend_with_regularisation(self) -> tuple:
        cost_function_history = []
        for iteration in range(self.__max_iter):
            self.__theta = self.__theta * (1 - self.alpha * (self.lmd / len(self.__X_train))) - (
                    self.alpha / len(self.__X_train) * self.__cost_function_derivative())
        cost_function_history.append(self.__cost_function_with_regularisation())
        return cost_function_history, self.__theta

    def predict(self, X_test):
        if self.__X_train is None or self.__y_train is None:
            raise AttributeError('Fit the training dataset to the model before training it!')
        if self.__regularisation:
            loss, self.__theta = self.__gradient_descend_with_regularisation()
        else:
            loss, self.__theta = self.__gradient_descend()
        return loss, self.__predict_probs(torch.cat((torch.ones((len(X_test), 1), dtype=torch.float64), X_test), 1))

    def __predict_probs(self, X_test):
        return torch.mm(X_test, self.__theta)

    def get_X_y(self):
        return self.__X_train, self.__y_train
