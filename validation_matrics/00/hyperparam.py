from itertools import product
import numpy as np
from validation import train_test_split, KFold, LeaveOneOut


class GridSearch:
    def __init__(self, estimator, param_grid, cv='kfold', cv_size=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.cv_size = cv_size

    def fit(self, X, y):
        #print('fit', X.shape, y.shape)
        combinations = product(*self.param_grid.values())
        self.params_ = {}
        for params in combinations:
            model = self.estimator(*params)
            if self.cv == 'train_test':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.cv_size)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
            elif self.cv == 'kfold':
                kfold = KFold(self.cv_size)
                scores = []
                for train, test in kfold.split(X):
                    #print('hyper', y.shape, train.shape)
                    model.fit(X[train], y[train])
                    scores.append(model.score(X[test], y[test]))
                score = np.array(scores).mean()
            elif self.cv == 'loocv':
                loocv = LeaveOneOut()
                scores = []
                for train, test in loocv.split(X):
                    model.fit(X[train], y[train])
                    scores.append(model.score(X[test], y[test]))
                score = np.array(scores).mean()
            self.params_[params] = score
        self.best_params_ = max(self.params_, key=lambda key: self.params_[key])
        self.best_estimator_ = self.estimator(*self.best_params_)

    def score(self, X, y):
        if not self.best_estimator_:
            self.fit(X, y)
        return self.best_estimator_.score(X, y)


class RandomSearch:
    def __init__(self, estimator, param_distributions, cv='kfold', cv_size=5, n_iter=10):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = cv
        self.cv_size = cv_size
        self.n_iter = n_iter

    def fit(self, X, y):
        self.params_ = {}
        for i in range(self.n_iter):
            params = []
            for param, values in self.param_distributions.items():
                if type(values) == dict:
                    dist_params = values['params']
                    if values['distribution'] == 'normal':
                        dist = np.random.normal
                    elif values['distribution'] == 'uniform':
                        dist = np.random.uniform
                    elif values['distribution'] == 'randint':
                        dist = np.random.randint
                    params.append(dist(*dist_params))
                else:
                    params.append(np.random.choice(values, 1))
            model = self.estimator(*params)
            if self.cv == 'train_test':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.cv_size)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
            elif self.cv == 'kfold':
                kfold = KFold(self.cv_size)
                scores = []
                for train, test in kfold.split(X):
                    model.fit(X[train], y[train])
                    scores.append(model.score(X[test], y[test]))
                score = np.array(scores).mean()
            elif self.cv == 'loocv':
                loocv = LeaveOneOut()
                scores = []
                for train, test in loocv.split(X):
                    model.fit(X[train], y[train])
                    scores.append(model.score(X[test], y[test]))
                score = np.array(scores).mean()
            self.params_[params] = score
        self.best_params_ = max(self.params_, key=lambda key: self.params_[key])
        self.best_estimator_ = self.estimator(*self.best_params_)

    def score(self, X, y):
        if not self.best_estimator_:
            self.fit(X, y)
        return self.best_estimator_.score(X, y)
