import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.len = X.shape[0]
        self.x_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.float32)
        self.dim = self.x_data.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def get_dim(self):
        return self.dim
