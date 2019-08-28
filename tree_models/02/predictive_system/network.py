import torch
from predictive_system.preprocess import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path


class Network(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def get_preprocessed_data():
    data = Dataset()
    data.get_text_features()
    data.encode_labels()
    data.encode_text()
    data.remove_outliers()
    data.encode_text()
    data.drop_columns()
    data.one_hot_encode()
    data.replace_nans()
    data.to_numpy()
    data.scale_data()
    return (data.data, data.target)


def train_model(train, target, save=True):

    X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=30, test_size=0.25)
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    model = Network(X_train.shape[1], 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    epochs = 3
    loss_func = torch.nn.MSELoss()

    for t in range(epochs):

        prediction = model(X_train)

        loss = loss_func(prediction, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction_valid = model(X_test)
        loss_valid = loss_func(prediction_valid, y_test)

        print(f"Epoch {t}: train - {torch.sqrt(loss)}; validation - {torch.sqrt(loss_valid)}")
    if save:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, Path('models') / 'checkpoint.pth')


def run():

    train, target = get_preprocessed_data()
    train_model(train, target)


if __name__ == '__main__':
    run()
