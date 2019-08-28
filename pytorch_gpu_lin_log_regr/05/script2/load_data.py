import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class HeartDataset(Dataset):
    def __init__(self, path):
        data = torch.tensor(pd.read_csv(path).values, dtype=torch.float32)
        self.len = data.shape[0]
        self.x_data = data[:, 0:12]
        self.y_data = data[:, [13]]
        self.dim = self.x_data.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def get_dim(self):
        return self.dim


class InsuranceDataset(Dataset):
    def __init__(self, path):
        data = torch.tensor(pd.get_dummies(
            pd.read_csv(path)).values, dtype=torch.float32)
        self.len = data.shape[0]
        self.x_data = data[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
        self.y_data = data[:, [3]]
        self.dim = self.x_data.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def get_dim(self):
        return self.dim
