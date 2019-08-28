from hw4.metrics import *


class LinearRegression(torch.nn.Module):
    def __init__(self, in_size, out_size, device='cpu'):
        super().__init__()
        self.linear = torch.nn.Linear(in_size, out_size).to(device)

    def forward(self, X):
        return self.linear(X)


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_size, out_size, device='cpu'):
        super().__init__()
        self.linear = torch.nn.Linear(in_size, out_size).to(device)

    def forward(self, X):
        return torch.sigmoid(self.linear(X))


def train_model(model, optimizer, criterion, train_loader, epochs, device):
    for epoch in range(epochs):

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
