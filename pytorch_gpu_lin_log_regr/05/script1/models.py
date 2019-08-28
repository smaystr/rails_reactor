import torch
from abc import ABC, abstractmethod

class BaseRegressor(ABC):
    def __init__(self, alpha: float=1e-1, iters: int=100, reg: str=None, C: float=1.0, batch_size: int=32, device: torch.device='cpu'):
        self.alpha = alpha
        self.iters = iters
        self.reg = reg
        self.C = C
        self.batch_size = batch_size
        self.device = device
        self.coef = None
    
    def fit(self, X, y):
        X, y, self.coef = self._preprocess_data(X, y)
        dataset_size = X.shape[0]
        
        for i in range(self.iters):
            for j in range(dataset_size // self.batch_size):
                X_batch = X[j * self.batch_size : j * self.batch_size + self.batch_size]
                y_batch = y[j * self.batch_size : j * self.batch_size + self.batch_size]
                self.coef -= (self.alpha/dataset_size) * (self._gradient(X_batch, y_batch) + self._reg_param())
        
        return self
    
    def predict(self, X):
        return self._desision_function(X)
    
    def _reg_param(self):
        if self.reg =='L1':
            return self.C * torch.sign(self.coef)
        elif self.reg == 'L2':
            return self.C * self.coef
        else:
            return 0.0
    
    def _preprocess_data(self, X, y):
        x = torch.cat([torch.ones((X.shape[0], 1), dtype=torch.float32, device=self.device), X], dim=1)
        coef = torch.zeros((x.shape[1], 1), dtype=torch.float32, device=self.device)
        
        return x, y, coef
    
    @abstractmethod
    def _desision_function(self, X):
        raise NotImplementedError
    
    @abstractmethod
    def _gradient(self, X, y):
        raise NotImplementedError


class LinearRegression(BaseRegressor):

    def __init__(self, alpha: float=1e-1, iters: int=100, reg: str=None, C: float=1.0, batch_size: int=32, device: torch.device='cpu'):
        super().__init__(alpha, iters, reg, C, batch_size, device)
    
    def _desision_function(self, X):
        return torch.cat([torch.ones((X.shape[0], 1), dtype=torch.float32, device=self.device), X], dim=1).mm(self.coef)
    
    def _gradient(self, X, y):
        return X.t().mm(X.mm(self.coef) - y)


class LogisticRegression(BaseRegressor):

    def __init__(self, alpha: float=1e-1, iters: int=100, reg: str=None, C: float=1.0, batch_size: int=32, device: torch.device='cpu'):
        super().__init__(alpha, iters, reg, C, batch_size, device)
    
    def predict_proba(self, X):
        proba = 1- self._sigmoid(torch.cat([torch.ones((X.shape[0], 1), dtype=torch.float32, device=self.device), X], dim=1).mm(self.coef))
        proba = torch.cat([proba, self._sigmoid(torch.cat([torch.ones((X.shape[0], 1), dtype=torch.float32, device=self.device), X], dim=1).mm(self.coef))], dim=1)
        return proba
    
    def _sigmoid(self, z):
        return 1/(1 + torch.exp(-z))
    
    def _desision_function(self, X):
        return self._sigmoid(torch.cat([torch.ones((X.shape[0], 1), dtype=torch.float32, device=self.device), X], dim=1).mm(self.coef)).round()
    
    def _gradient(self, X, y):
        return X.t().mm(self._sigmoid(X.mm(self.coef)) - y)
    
