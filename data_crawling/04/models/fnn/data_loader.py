import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class ApartmentsDataset(Dataset):
    def __init__(self, path):
        data = pd.get_dummies(pd.read_csv(path))
        x, y = data.drop('price', axis=1), pd.DataFrame(data['price'])
        self.len = data.shape[0]
        self.x_data = torch.tensor(x.values.astype(np.float32), dtype=torch.float32)
        self.y_data = torch.tensor(y.values, dtype=torch.float32)
        self.dim = self.x_data.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def get_dim(self):
        return self.dim