import numpy as np

def train_test_split(x_data, y_data, test_size=0.1, shuffle=False):
    test_dim = int(test_size * x_data.shape[0])
    train_dim = x_data.shape[0] - test_dim 
    if shuffle:
        train_indexes = np.random.randint(x.shape[0], size=train_dim, replace=False)
    else:
        train_indexes = np.arange(train_dim)
    test_indexes = np.delete(np.arange(x_data.shape[0]), train_indexes)
    return (x_data[train_indexes],np.expand_dims(y_data[train_indexes], axis=1),  
            x_data[test_indexes], np.expand_dims(y_data[test_indexes], axis=1))

class KFold:
    def __init__(self, number_folds=5, shuffle=False):
        if number_folds < 2:
            raise Exception(f"Minimum 2 folds needed. {number_folds} passed.")
        self.number_folds = int(number_folds)
        self.shuffle = shuffle

    def split(self, data):
        indicies = np.arange(0, data.shape[0])
        if self.shuffle:
            np.random.shuffle(indicies)
        validation_size = int(data.shape[0] * 1 / self.number_folds)
        k_folds_validation_splits = [ indicies[indices * validation_size : (indices + 1) * validation_size] for indices in range(self.number_folds)]
        return [(indicies, indicies[~i]) for i in k_folds_validation_splits]

class LeaveOneOut:
    def __init__(self, shuffle=True):
        self.shuffle = shuffle
        
    def split(self, data):
        indexes = np.arange(data.shape[0])
        if self.shuffle:
            np.random.shuffle(indexes)
        return [(np.delete(indexes, i), [i]) for i in indexes]
