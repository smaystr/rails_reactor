import torch
from torch.nn import Module


class LinearRegression(Module):
    def __init__(self, input_size, output_size, device='cpu'):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn \
            .Linear(input_size, output_size) \
            .to(device) \
            .double()
        self.linear = self.linear \
            .type(torch.double) \
            .to(device)

    def forward(self, X):
        prediction = self.linear(X)
        return prediction


class LogisticRegression(Module):
    def __init__(self, input_size, output_size, device='cpu'):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn \
            .Linear(input_size, output_size) \
            .to(device) \
            .double()
        self.linear = self.linear \
            .type(torch.double) \
            .to(device)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        prediction = self.sigmoid(self.linear(X))
        return prediction
