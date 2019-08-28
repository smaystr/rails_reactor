import numpy as np
import pandas as pd

def normalize(X: np.ndarray):
    norm = np.copy(X)
    mean = norm.mean(axis=0)
    std = norm.std(axis=0)
    return (norm - mean)/std

def load_data(path: str, type: str):
    if type == 'linear':
        return linear_data(path)
    elif type == 'logistic':
        return logistic_data(path)

def linear_data(path: str):
    csv = pd.read_csv(path)
    data = pd.get_dummies(csv).to_numpy()

    X = data[:, [0,1,2,4,5,6,7,8,9,10,11]]
    y = data[:, [3]]

    return X, y

def logistic_data(path: str):
    csv = pd.read_csv(path)
    data = csv.to_numpy()

    X = data[:, 0:12]
    y = data[:, [13]]

    return X, y
