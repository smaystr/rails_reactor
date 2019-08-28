import numpy as np
from utils import prepare_data


class LinearRegression:

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, x_train, y_train):
        x_train, y_train = prepare_data(x_train, y_train, self.fit_intercept)
        weights = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        if self.fit_intercept:
            self.coef_ = weights[1:]
            self.intercept_ = weights[0]
        else:
            self.coef_ = weights
            self.intercept_ = .0
        return self

    def predict(self, x):
        return np.dot(x, self.coef_) + self.intercept_
