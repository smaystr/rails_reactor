import numpy as np

class LogisticRegression:

    def __init__(self, alpha=1e-1: float, iters=100: int, reg=None: str, C=1.0: float):
        self.alpha = alpha
        self.iters = iters
        self.reg = reg
        self.C = C
        self.coef = None

    def fit(self, X, y):
        """
        Fit linear model
        Parameters:
        X: 2d-array
        y: 2d-array (vector)
        Returns: self
        """
        X, y, self.coef = self._preprocess_data(X, y)
        dataset_size = X.shape[0]

        for i in range(self.iters):
            self.coef -= (self.alpha/dataset_size) * (X.transpose().dot(self._sigmoid(X.dot(self.coef)) - y) + self._reg_param())

        return self
    
    def predict(self, X):
        return self._desision_function(X)
    
    def predict_proba(self, X):
        """
        Predicts not classes but probability of belongin to the class
        Parameters:
        X: 2d-array
        Returns: 2d-array
        """
        proba = 1- self._sigmoid(np.insert(X, 0, 1, axis=1).dot(self.coef))
        proba = np.append(proba, self._sigmoid(np.insert(X, 0, 1, axis=1).dot(self.coef)), axis=1)
        return proba
    
    def _reg_param(self):
        """
        Return regularization parameter for ech type of regularization
        """
        if self.reg =='L1':
            return self.C * np.sign(self.coef)
        elif self.reg == 'L2':
            return self.C * self.coef
        else:
            return 0.0
    
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def _desision_function(self, X):
        return self._sigmoid(np.insert(X, 0, 1, axis=1).dot(self.coef)).round()
    
    def _preprocess_data(self, X, y):
        """
        Creates array of coefficients
        Parameters:
        X: 2d-array
        y: 2d-array (vector)
        """
        x = np.insert(X, 0, 1, axis=1)
        coef = np.zeros((x.shape[1], 1))

        return x, y, coef
