import pandas as pd
import torch

def normalize(X):
    norm = X.clone()
    mean = norm.mean(dim=0)
    std = norm.std(dim=0, unbiased=False)
    return (norm - mean)/std

def load_data(path: str, type: str, device: torch.device):
    if type == 'linear':
        return linear_data(path, device)
    elif type == 'logistic':
        return logistic_data(path, device)

def linear_data(path: str, device: torch.device):
    csv = pd.read_csv(path)
    data = torch.tensor(pd.get_dummies(csv).values, dtype=torch.float32, device=device)

    X = data[:, [0,1,2,4,5,6,7,8,9,10,11]]
    y = data[:, [3]]

    return X, y

def logistic_data(path: str, device: torch.device):
    csv = pd.read_csv(path)
    data = torch.tensor(csv.values, dtype=torch.float32, device=device)

    X = data[:, 0:12]
    y = data[:, [13]]

    return X, y