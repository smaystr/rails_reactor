import numpy as np


class MinMaxScalar:

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)

    def transform(self, X):
        np.seterr(divide='ignore', invalid='ignore')

        return np.nan_to_num(
            (X - self.min) / (self.max - self.min)
            )

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
