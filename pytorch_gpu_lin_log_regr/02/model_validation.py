import time

import numpy as np
import torch


def train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        num_epochs,
        metric,
        device,
        print_step
):
    """
    Training the model
    :type model: torch.nn.Module
    :type criterion: torch.nn
    :type optimizer: torch.optim.Optimizer
    :type train_loader: torch.utils.data.DataLoader
    :type num_epochs: int
    :param metric: method
    :type device: torch.device
    :type print_step: int
    """
    stats = {}
    loss_values = []
    metric_values = []
    batches_benchmarks = []
    loss = 0
    for epoch in range(num_epochs):
        prediction, target = 0, 0
        working_time = 0
        for local_batch, local_targets in train_loader:
            stopwatch = time.time()
            data = local_batch \
                .to(device) \
                .type(torch.DoubleTensor)
            target = local_targets \
                .to(device) \
                .type(torch.DoubleTensor)
            prediction = model(data)
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            working_time = time.time() - stopwatch
        if epoch % print_step == 0:
            print(f'Epoch {epoch}')
            print(f'{criterion.__class__.__name__}: {loss.item()}')
            print('Working out: %.5f micro seconds' % working_time)
            loss_values.append(loss.item())
            prediction = prediction \
                .cpu() \
                .detach() \
                .numpy() \
                .round()
            target = target \
                .cpu() \
                .detach() \
                .numpy()
            metric_values.append(metric(target, prediction))
            batches_benchmarks.append(working_time)
    stats['loss_values'] = loss_values
    stats['metric_values'] = metric_values
    stats['optimizer'] = optimizer.__class__.__name__
    stats['metric'] = metric.__class__.__name__
    stats['num_epochs'] = int(num_epochs / print_step)
    stats['device'] = device
    stats['print_step'] = print_step
    stats['working_time'] = batches_benchmarks
    return stats


def test_model(
        model,
        loader,
        metric,
        device,
        isValidation=False
):
    """
    Training the model
    :type model: torch.nn.Module
    :type loader: torch.utils.data.DataLoader
    :param metric: method
    :type device: torch.device
    :type isValidation: bool
    """
    if isValidation:
        print('Validating the model!')
    else:
        print('Testing the model!')
    predictions = []
    targets = []
    for local_data, local_targets in loader:
        data = local_data \
            .to(device) \
            .type(torch.DoubleTensor)
        target = local_targets \
            .to(device) \
            .type(torch.DoubleTensor)
        prediction = model(data)
        predictions.extend(prediction)
        targets.extend(target)
    predictions = np.array([
        prediction
            .cpu()
            .detach()
            .numpy()
            .round()
        for prediction in predictions
    ])
    targets = np.array([
        target
            .cpu()
            .detach()
            .numpy()
        for target in targets
    ])
    return predictions, metric(targets, predictions)
