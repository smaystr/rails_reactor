import numpy as np

class GridSearchCV:
    
    max_iter = [100, 1000, 10000]
    learning_rate = [0.001, 0.01, 0.1, 1]

    def __init__(self, estimator, scoring):
        self.estimator = estimator
        self.scoring = scoring
    
    def evaluate_model(self, x_data, y_data):
        res = []
        for iterations in max_iter:
            for rate in learning_rate:
                result = self.estimator(learning_rate=rate, num_iterations=iterations).fit(x_data, y_data)
                score = self.scoring(self.estimator.predict(x_data), y_data)
                print('iterations: {iterations}; learning_rate: {rate}; score: {score}')
                res[result] = (iterations, rate)
        return res

class RandomSearchCV:
    max_iter = np.random.randint(10000, 1000000, 3)
    learning_rate = np.random.uniform(0.0001, 0.1, 4)

    def __init__(self, estimator, scoring):
        self.estimator = estimator
        self.scoring = scoring
    
    def evaluate_model(self, x_data, y_data):
        res = []
        for iterations in max_iter:
            for rate in learning_rate:
                result = self.estimator(learning_rate=rate, num_iterations=iterations).fit(x_data, y_data)
                score = self.scoring(self.estimator.predict(x_data), y_data)
                print('iterations: {iterations}; learning_rate: {rate}; score: {score}')
                res[result] = (iterations, rate)
        return res