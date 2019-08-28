import torch


class LinearRegression(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = torch.nn.Linear(in_size, out_size)

    def forward(self, X):
        lin = self.linear(X)
        return lin


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = torch.nn.Linear(in_size, out_size)

    def forward(self, X):
        lin = self.linear(X)
        return torch.sigmoid(lin)
