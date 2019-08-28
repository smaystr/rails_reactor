import torch
import time
import tensorboardX
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dimensions, output_dimensions, device='cpu'):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dimensions, output_dimensions).to(device)

    def forward(self, x):
        preds = self.linear(x)
        return preds


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dimensions, output_dimensions, device='cpu'):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dimensions, output_dimensions).to(device)

    def forward(self, x):
        preds = self.linear(x)
        return torch.sigmoid(preds)


def train_model(model, learning_rate, optimizer, epochs, loader, device):
    writer = tensorboardX.SummaryWriter()
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    if isinstance(model, LinearRegression):
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.BCELoss()

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
                writer.add_scalar('mean_squared_error', mean_squared_error(labels, outputs), epoch)
                writer.add_scalar('mean_absolute_error', mean_absolute_error(labels, outputs), epoch)
                writer.add_scalar('r2_score', r2_score(labels, outputs), epoch)
            else:
                writer.add_scalar('accuracy_score', accuracy_score(labels, outputs), epoch)
                writer.add_scalar('recall_score', recall_score(labels, outputs), epoch)
                writer.add_scalar('precision_score', precision_score(labels, outputs), epoch)
                writer.add_scalar('f1_score', f1_score(labels, outputs), epoch)

    writer.close()

def validate_model(model, loader, device):
    if isinstance(model, LinearRegression):
        criterion = torch.nn.MSELoss() 
    else:
        criterion = torch.nn.BCELoss()

    loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
    model.train()
    return loss
