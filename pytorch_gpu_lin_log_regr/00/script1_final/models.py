import torch

class LinearRegression:
    def __init__(self, max_iter = 1000, C = 1.0, batch_size = 32, device: torch.device ='cpu', regularization = 'l1', 
                 learning_rate = 0.01):
        self.max_iter = max_iter
        self.C = C
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.regularization = regularization
    
    def regularize(self):
        if self.regularization =='l1':
            return self.C * torch.abs(self.theta)
        else:
            return self.C * self.theta

    def fit(self, X_data, y_data):
        self.size = len(X_data)
        X_data = torch.tensor(X_data, dtype = torch.float, device = self.device)
        y_data = torch.tensor(y_data, dtype = torch.float, device = self.device)
        self.theta = torch.rand((X_data.shape[1], 1), dtype = torch.float, device = self.device)
                 
        for i in range(self.max_iter):
            for j in range(self.size // self.batch_size):
                X_batch = X_data[j * self.batch_size : (j + 1) * self.batch_size]
                y_batch = y_data[j * self.batch_size : (j + 1) * self.batch_size]
                prediction = X_batch.t().mm(self.theta)
                error = prediction - y_batch
                gradient = X_batch.t().mm(error)
                self.theta -= (self.learning_rate / self.size) * gradient + self.regularize()
        return self

    def predict(self, X_data):
        X_data = torch.tensor(X_data, dtype = torch.float, device = self.device)
        return X_data.mm(self.theta)

class LogisticRegression:
    def __init__(self, max_iter = 1000, C = 1.0, batch_size = 32, device: torch.device='cpu', regularization = 'l1',
                learning_rate = 0.01):
        self.max_iter = max_iter
        self.C = C
        self.batch_size = batch_size
        self.device = device
        self.regularization = regularization
        
    def regularize(self):
        if self.regularization == 'l1':
            return self.C * torch.abs(self.theta)
        else:
            return self.C * self.theta * self.theta
    
    def fit(self, X_data, y_data):
        self.size = len(X_data)
        X_data = torch.tensor(X_data, dtype = torch.float, device = self.device)
        y_data = torch.tensor(y_data, dtype = torch.float, device = self.device)
        self.theta = torch.rand((X_data.shape[1], 1), dtype = torch.float, device = self.device)
        
        for i in range(self.max_iter):
            for i in range(self.size // self.batch_size):
                X_batch = X_data[j * self.batch_size : (j + 1) * self.batch_size]
                y_batch = y_data[j * self.batch_size : (j + 1) * slef.batch_size]
                gradient = X_batch.t().mm(self.sigmoid(X_batch.mm(self.theta)) - y_batch)
                self.theta -= (self.learning_rate / self.size) * gradient + self.regularize()
        return self

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z)) 
    
    def predict_proba(self, X_data):
        X_data = torch.tensor(X_data, dtype = torhc.float, device = self.device)
        return X_data.t().mm(self.theta)
    
    def predict(self, X_data):
        return self.predict_proba(X_data) >= 0.5