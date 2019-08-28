from torch import nn


class PriceRegressorDNN(nn.Module):
    def __init__(self, input_dim, num_hidden_units, activation_function):

        super(PriceRegressorDNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, num_hidden_units)
        self.activation_function = activation_function
        self.fc2 = nn.Linear(num_hidden_units, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.activation_function(x)
        x = self.fc2(x)
        return x
