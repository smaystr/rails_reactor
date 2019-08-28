import torch
import time
import tensorboardX
from hw6_draft.metrics import *


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).to(device)

    def forward(self, x):
        out = self.linear(x)
        return out


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).to(device)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


def train_model(model, alpha, optim, epochs, loader, device):
    writer = tensorboardX.SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha) if optim == "adam" else torch.optim.SGD(model.parameters(), lr=alpha)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    criterion = torch.nn.MSELoss() if isinstance(model, LinearRegression) else torch.nn.BCELoss()

    for epoch in range(epochs):
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            scheduler.step(epoch)

            if isinstance(model, LinearRegression):
                writer.add_scalar('mse', mse(labels, outputs), epoch)
                writer.add_scalar('rmse', rmse(labels, outputs), epoch)
                writer.add_scalar('mae', mae(labels, outputs), epoch)
                writer.add_scalar('r2', r2(labels, outputs), epoch)
            else:
                writer.add_scalar('accuracy', accuracy(labels, outputs), epoch)
                writer.add_scalar('recall', recall(labels, outputs), epoch)
                writer.add_scalar('precision', precision(labels, outputs), epoch)
                writer.add_scalar('f1', f1(labels, outputs), epoch)

    writer.close()

def validate_model(model, loader, device):
    criterion = torch.nn.MSELoss() if isinstance(model, LinearRegression) else torch.nn.BCELoss()

    loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss += criterion(outputs, labels).item()

    model.train()
    return loss
