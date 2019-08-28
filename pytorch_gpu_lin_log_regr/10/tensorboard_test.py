from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch
import utils

from pytorch_api import LinearRegression, to_torch


def main():

    use_cuda = torch.cuda.is_available()
    print('cuda:', use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    epochs = 1000
    optimizers = [optim.SGD, optim.Adam]
    batch_size_list = [100, 1000, 10000]
    lr_list = [1e-2, 1e-3, 1e-4]

    criterion = torch.nn.MSELoss()
    train, test = utils.download_train_test(
        'insurance_train.csv', 'insurance_test.csv', url=utils.URL_DATA)

    X_train, y_train, X_test, y_test = utils.preprocess_medicalcost(
        train, test)
    X_train, y_train = to_torch(X_train).float(), to_torch(y_train)

    n_features = X_train.shape[1]

    opt = optimizers[0]

    # fixed values in case of iteration over one option
    batch_size = 100
    lr = 1e-3
    opt = optimizers[0]

    # choose line for iteration or use several iterators

    # for opt in optimizers: 
    for batch_size in batch_size_list:
        for lr in lr_list:
            model = LinearRegression(n_features).to(device)
            trainloader = Data.DataLoader(
                dataset=TensorDataset(X_train, y_train),
                batch_size=batch_size)

            optimizer = opt(model.parameters(), lr=lr)

            comment=f' optimizer {opt} batch_size={batch_size} lr={lr}'
            tb = SummaryWriter(comment=comment)

            for epoch in range(epochs):
                running_loss = 0.0
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                tb.add_scalar('Loss', running_loss, epoch)

            tb.close()

if __name__ == '__main__':
    main()