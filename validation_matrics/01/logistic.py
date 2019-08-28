import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=100000, fit_intercept=True, verbose=False, regularization="l2", C = 1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.C = C
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def regularize(self):
        if self.regularization == "l1":
            return self.theta / (self.C * self.size * np.abs(self.weights))
        return 1 / (self.C * self.size) * self.theta
    
    def loss(self, h, y):
        log_pos = (-y * np.log(h))
        log_neg = (-(1-y) * np.log(1-h))
        return (log_pos + log_neg).mean()
    
    def fit(self, X, y):
        self.size = len(X)
        if self.fit_intercept:
            X = self.__add_intercept(X)
        # weights initialization
        self.theta = np.random.rand(X.shape[1])
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / self.size  + self.regularize()
            self.theta -= self.learning_rate * gradient
            
            if(self.verbose == True and i % 1000 == 0):
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                print(f'loss: {self.loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    