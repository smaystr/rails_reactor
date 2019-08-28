from __future__ import annotations
import numpy as np
import metrics

class LogisticRegression():
    '''
    Logistic Regression class implements logistic regression with gradient descent.
    '''

    def __init__(self, lr: float = 1e-4, epoch: int = 100, penalty: str = 'l2', C: float = 1.0):
        self.lr = lr
        self.epoch = epoch
        self.penalty = penalty
        self.C = C

        self.metrics = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f_score': metrics.f_score
        }

    def __sigmoid__(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y:np.ndarray) -> LogisticRegression:
        '''
        Train model on data.
        '''
        self.w = np.zeros((X.shape[1] + 1, 1), dtype=np.float32)
        m = X.shape[0]
        y_ = y.reshape((-1, 1))
        X_ = np.ones((X.shape[0], X.shape[1] + 1))
        X_[:, 1:] = X

        for i in range(self.epoch):
            predicted = self.predict_proba(X)
            e = X_.transpose().dot(predicted - y_)

            if self.penalty == 'l1':
                reg_p = self.C * np.sign(self.w)
            elif self.penalty == 'l2':
                reg_p = self.C * self.w
            else:
                reg_p = 0

            self.w = self.w - self.lr * (e + reg_p) / m

        return self

    def predict(self, X: np.ndarray, thr: float = .5) -> np.ndarray:
        '''
        Predict labels for data.
        '''
        return (self.predict_proba(X) > thr).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict probabilities for classes.
        '''
        X_ = np.ones((X.shape[0], X.shape[1] + 1))
        X_[:, 1:] = X
        return self.__sigmoid__(X_.dot(self.w))

    def score(self, X: np.ndarray, y: np.ndarray, thr: float = .5, metric: str = 'accuracy') -> float:
        '''
        Model scoring.
        '''
        y_ = y.reshape((-1, 1))
        return self.metrics.get(metric, 'accuracy')(self.predict(X, thr), y_)


class LinearRegression():
    '''
    Linear Regression class implements linear regression with gradient descent.
    '''

    def __init__(self, lr: float = 1e-4, epoch: int = 100, penalty: str = 'l2', C: float = 1.0):
        self.lr = lr
        self.epoch = epoch
        self.penalty = penalty
        self.C = C

        self.metrics = {
            'mse': metrics.mse,
            'rmse': metrics.rmse,
            'mae': metrics.mae
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        '''
        Train model on data.
        '''
        self.w = np.zeros((X.shape[1] + 1, 1), dtype=np.float32)
        m = X.shape[0]
        y_ = y.reshape((-1, 1))
        X_ = np.ones((X.shape[0], X.shape[1] + 1))
        X_[:, 1:] = X

        for i in range(self.epoch):
            predicted = self.predict(X)
            e = X_.transpose().dot(predicted - y_)

            if self.penalty == 'l1':
                reg_p = self.C * np.sign(self.w)
            elif self.penalty == 'l2':
                reg_p = self.C * self.w
            else:
                reg_p = 0

            self.w = self.w - self.lr * (e + reg_p) / m
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict target.
        '''
        X_ = np.ones((X.shape[0], X.shape[1] + 1))
        X_[:, 1:] = X
        return X_.dot(self.w)

    def score(self, X: np.ndarray, y: np.ndarray, metric: str = 'rmse') -> float:
        '''
        Model scoring.
        '''
        y_ = y.reshape((-1, 1))
        return self.metrics.get(metric, 'rmse')(self.predict(X), y_)
