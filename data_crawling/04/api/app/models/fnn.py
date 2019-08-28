import torch


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
        for i in range(len(self.hidden)):
            out = self.hidden[i](out)

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
