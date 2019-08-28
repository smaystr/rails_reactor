import torch
import time
from typing import Union, Tuple
from torch.autograd import Variable

from models.linear_regression import LinearReg, TorchLinearReg
from models.logistic_regression import LogisticReg, TorchLogisticReg


def script_1(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    task: int,
    model_params: dict,
    device: torch.device,
) -> Tuple[Union[LinearReg, LogisticReg], str]:
    model_class = LogisticReg if task == 1 else LinearReg
    model = model_class(device=device, **model_params)
    model.fit(X_train, y_train)
    model_report = (
        f"**Device**: *{device}*\n\n"
        f"**Fitting time**: *{model.fit_time_}s*\n\n"
        f"**Weights**:\n```\n{model.get_thetas()}\n```"
    )
    return model, model_report


def script_2(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    task: int,
    model_params: dict,
    device: torch.device,
) -> Tuple[Union[TorchLinearReg, TorchLogisticReg], str]:
    model_class = TorchLogisticReg if task == 1 else TorchLinearReg
    criterion = torch.nn.BCELoss() if task == 1 else torch.nn.MSELoss()

    torch.manual_seed(0)
    num_examples = X_train.shape[0]
    input_dim = X_train.shape[1]
    learning_rate = 0.01
    epochs = 5
    X_train = Variable(X_train.float())
    y_train = Variable(y_train)

    model = model_class(input_dim).to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    start = time.time()
    for epoch in range(epochs):
        for row_idx in range(num_examples):
            _X = X_train[row_idx]
            _y = y_train[row_idx]
            y_pred = model(_X)[0]

            loss = criterion(y_pred, _y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    fitting_time = time.time() - start

    model_report = (
        f"**Device**: *{device}*\n\n"
        f"**Fitting time**: *{fitting_time}s*"
    )
    return model, model_report
