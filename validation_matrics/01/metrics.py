import numpy as np

class Metrics :
    def mean_squared_error(true, predicted):
        return np.mean(np.square(true - predicted))
    
    def mean_absolute_error(true, predicted):
        return np.mean(np.abs(true - predicted))
    
    def root_mean_squared_error(true, predicted):
        return np.sqrt(mean_squared_error(true, predicted))
    
    def mean_absolute_percentge_error(true, predicted):
        return np.mean(np.abs(true-predicted)/true)*100
    
    def mean_percentage_error(true, predicted):
        return np.mean((true-predicted)/true) * 100
    
    def accuracy(true, predicted):
        return np.sum(true == predicted)/len(true)
    
    def precision(true, predicted):
        true_positive = np.sum(np.logical_and(true == 1, predicted == 1))
        false_positive = np.sum(np.logical_and(true == 1, predicted == 0 ))
        return true_positive /(true_positive + false_positive)
    
    def recall(true, predicted):
        true_positive = np.sum(np.logical_and(true == 1, predicted == 1))
        false_negative = np.sum(np.logical_and(true == 0, predicted == 1))
        return true_positive/(true_positive + false_negative)
    
    def f_1_score(true, predicted):
        top = precision(true, predicted)*recall(true, predicted)
        botom = precision(true, predicted) + recall(true, predicted)
        return 2*top/botom