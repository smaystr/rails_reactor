import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class FNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, activation_function='relu'):
        super(FNN, self).__init__()
        self.fc = torch.nn.Linear(input_dim, hidden_dim)
        self.activ = self.get_activation(activation_function)
        self.hidden = torch.nn.ModuleList()
        for i in range(hidden_num-1):
            self.hidden.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.hidden.append(self.get_activation(activation_function))
        
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.fc(x)
        out = self.activ(out)
        for layer in self.hidden:
            out = layer(out)

        out = self.out(out)

        return out

    def get_activation(self, activ_name):
        if activ_name == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activ_name == 'tanh':
            return torch.nn.Tanh()
        elif activ_name == 'relu':
            return torch.nn.ReLU()
        elif activ_name == 'leaky':
            return torch.nn.LeakyReLU()
        elif activ_name == 'elu':
            return torch.nn.ELU()


def train_nn(model, alpha, epochs, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        for i, data in enumerate(loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def validate_nn(model, train_loader, test_loader):
    mae_train, mae_test = [], []
    mse_train, mse_test = [], []
    r2_train, r2_test = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            mae_train.append(mean_absolute_error(labels, outputs))
            mse_train.append(mean_squared_error(labels, outputs))
            r2_train.append(r2_score(labels, outputs))

        for inputs, labels in test_loader:
            outputs = model(inputs)
            mae_test.append(mean_absolute_error(labels, outputs))
            mse_test.append(mean_squared_error(labels, outputs))
            r2_test.append(r2_score(labels, outputs))

    model.train()

    return mae_train, mae_test, mse_train, mse_test, r2_train, r2_test
