class StandardScaler():
    def __init__(self, with_std=True, with_mean=True):
        self.with_std = with_std
        self.with_mean = with_mean
        self.mean = 0
        self.std = 1

    def fit(self, X):
        if self.with_std:
            self.std = X.std(axis=0)
        if self.with_mean:
            self.mean = X.mean(axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
