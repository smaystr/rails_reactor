import numpy as np
from pathlib import Path
from metrics import accuracy, precision, recall, f1, mse, rmse, mae


def _add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


class LogisticRegression:
    # fit_intercept is True by default and isn't tunable, because it will lower over metrics a lot
    def __init__(self, lr=1e-4, num_iter=1000, penalty='l2', C=1.0, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.penalty = penalty
        self.C = C
        self.verbose = verbose

        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        self.weights = np.random.rand((X.shape[1] + 1, 1))
        y_true = y.reshape((-1, 1))
        X_ = _add_intercept(X)

        for i in range(self.num_iter):
            reg_p = 0
            predicted = self.predict_proba(X)
            grad = X_.T.dot(predicted - y_true)

            if self.penalty == 'l1':
                reg_p = self.C * np.sign(self.weights)
            elif self.penalty == 'l2':
                reg_p = self.C * np.power(self.weights, 2)

            self.weights -= self.lr * (grad + reg_p) / X.shape[0]

            if self.verbose and i % 10000 == 0:
                print(f'loss: {self.loss(X, y)} \t')

    def predict_proba(self, X):
        X = _add_intercept(X)
        return self.__sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=.55):
        return self.predict_proba(X) >= threshold

    def score(self, X, y, thr=.5, metric='accuracy'):
        return self.metrics.get(metric, 'accuracy')(self.predict(X, thr), y.reshape(-1, 1))

    def loss(self, X, y):
        predicted = self.predict_proba(X)
        y_true = y.reshape(-1, 1)
        return self.__loss(predicted, y_true)

    def params(self):
        return {
            "lr": self.lr,
            "epoch": self.num_iter,
            "penalty": self.penalty,
            "C": self.C,
        }

    def serialize(self, path: str):
        model_string = "LOGIT\n"
        for k, v in self.params().items():
            model_string += f"{k}: {v}\n"
        model_string += f"{self.weights.flatten()}\n"

        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "output.model").write_text(model_string, "utf-8")

    def deserialize(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().split("\n")
        if content[0] != "LOGIT":
            raise RuntimeError("Check your model type! It is wrong")
        for v in content[1:-2]:
            des = v.split(": ")
            if des[0] == "lr":
                self.C = float(des[1])
            elif des[0] == "epoch":
                self.num_iter = int(des[1])
            elif des[0] == "penalty":
                self.penalty = des[1]
            elif des[0] == "C":
                self.C = float(des[1])
        self.weights = np.fromstring(
            content[-2].replace("[", "").replace("]", ""), sep=" "
        ).reshape((-1, 1))
        return self


class LinearRegression:
    def __init__(self, lr=1e-4, num_iter=100, penalty='l2', C=1.0):
        self.lr = lr
        self.num_iter = num_iter
        self.penalty = penalty
        self.C = C

        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

    def fit(self, X, y):
        self.weights = np.random.rand((X.shape[1] + 1, 1))
        y_true = y.reshape((-1, 1))
        X_ = _add_intercept(X)

        for i in range(self.num_iter):
            reg_p = 0
            predicted = self.predict(X)
            grad = X_.T.dot(predicted - y_true)

            if self.penalty == 'l1':
                reg_p = self.C * np.sign(self.weights)
            elif self.penalty == 'l2':
                reg_p = self.C * np.power(self.weights, 2)

            self.weights -= self.lr * (grad + reg_p) / X.shape[0]

    def predict(self, X):
        X_ = _add_intercept(X)
        return X_.dot(self.weights)

    def loss(self, X, y):
        predicted = self.predict(X)
        return np.mean(np.square(predicted, y.reshape(-1, 1)))

    def score(self, X, y, metric='rmse'):
        return self.metrics.get(metric, 'rmse')(self.predict(X), y.reshape((-1, 1)))

    def params(self):
        return {
            "lr": self.lr,
            "epoch": self.num_iter,
            "penalty": self.penalty,
            "C": self.C,
        }

    def serialize(self, path: str):
        model_string = "LINREG\n"
        for k, v in self.params().items():
            model_string += f"{k}: {v}\n"
        model_string += f"{self.weights}\n"

        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True, mode=0o755)
        (dir_path / "output.model").write_text(model_string, "utf-8")

    def deserialize(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().split("\n")
        if content[0] != "LINREG":
            raise RuntimeError("Wrong model type.")
        for v in content[1:-2]:
            des = v.split(": ")
            if des[0] == "lr":
                self.C = float(des[1])
            elif des[0] == "epoch":
                self.num_iter = int(des[1])
            elif des[0] == "penalty":
                self.penalty = des[1]
            elif des[0] == "C":
                self.C = float(des[1])
        self.weights = np.fromstring(
            content[-2].replace("[", "").replace("]", ""), sep=" "
        ).reshape((-1, 1))
        return self

