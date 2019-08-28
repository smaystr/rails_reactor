import numpy as np

class LinearRegression():
    def __init__(self, learning_rate = 0.01, num_iterations=10000, fit_intercept = True, verbose=False, error=1e-1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.error = error

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
        
    def cost(self, X, y):
        return (np.square((np.dot(X, self.theta) - y))/(2*self.size)
    
    def fit(self, X, y):
        self.size = len(X)
        if self.fit_intercept:
            X = self.__add_intercept(X)
        # weights initialization
        self.theta = np.random.rand(X.shape[1])
        
        for i in range(self.num_iterations):
            error_before_iteration = self.cost(X, y)
            prediction = np.dot(X, self.theta)
            error = prediction - y
            gradient =  np.dot(X.T, error)/self.size + self.regularize()
            self.theta -= self.learning_rate * gradient 
            error_after_iteration = self.cost(X, y)
            diff = np.abs(np.sum(error_before_iteration - error_after_iteration))
            if (self.verbose and i % 100 == 0):
                print('{} {}'.format(i, np.sum(error_after_iteration)))
            if diff <= self.error:
                break
        
    def predict(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return np.dot(X,self.theta)