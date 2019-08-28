import torch.nn as nn
from torch import sigmoid
import time
from matplotlib import pyplot as plt
import seaborn as sns
import pathlib
import torch
from torch.utils.data import DataLoader, TensorDataset
from settings import REPORT_PATH
from utils import load_config

config = load_config()


class LinearModel(nn.Module):
    def __init__(self, num_features, apply_sigmoid=False):

        super(LinearModel, self).__init__()
        self.fc_layer = nn.Linear(num_features, 1)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        outputs = self.fc_layer(x)
        if self.apply_sigmoid:
            outputs = sigmoid(outputs)
        return outputs


def create_loaders(X_train, X_test, Y_train, Y_test):

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=config.get("BATCH_SIZE"), shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.get("BATCH_SIZE"), shuffle=False
    )

    return train_loader, test_loader


def train_model(
    model, optimizer, criterion, train_loader, num_epochs, metric, print_epoch=100
):

    training_stats = dict()

    losses = []
    metrics = []
    training_time = []

    for n in range(num_epochs):

        time_start = time.time()

        moving_loss, moving_metric = 0, 0

        for i, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            metric_val = metric(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            moving_loss += loss.item()
            moving_metric += metric_val.item()
        batch_time = time.time() - time_start

        losses.append(moving_loss / len(train_loader))
        metrics.append(moving_metric / len(train_loader))
        training_time.append(batch_time)

        if n % print_epoch == 0:
            print(f"Epoch {n + 1} \t")
            print(
                f"{criterion.__class__.__name__} is {moving_loss / len(train_loader):.4f} \t"
            )
            print(f"{metric.__name__} is {moving_metric / len(train_loader):.4f}")
            print(f"Training time for epoch is {batch_time:.4f}")
            moving_loss, moving_metric = 0.0, 0.0

    training_stats[f"{criterion.__class__.__name__}"] = losses
    training_stats[f"{metric.__name__}"] = metrics
    training_stats["seconds_per_batch"] = training_time
    training_stats["num_epochs"] = num_epochs
    training_stats["hyperparams"] = {
        "lr": optimizer.defaults["lr"],
        "batch_size": train_loader.batch_size,
        "optimizer": optimizer.__class__.__name__,
    }

    return training_stats


def test_model(trained_model, test_loader, metric):

    outputs_arr = []
    labels_arr = []

    for i, batch in enumerate(test_loader):
        inputs, labels = batch
        outputs = trained_model(inputs)

        outputs_arr.extend(outputs)
        labels_arr.extend(labels)

    predictions = torch.cat(outputs_arr)
    labels_arr = torch.cat(labels_arr)

    metric_val = metric(predictions, labels_arr)
    return predictions, metric_val


def generate_report(stats_dict):

    colors = ["coral", "red", "blue"]

    markdown_path = pathlib.Path(REPORT_PATH)
    pics_path = pathlib.Path("./figures")

    pics_path.mkdir(exist_ok=True)

    with markdown_path.open("a") as file:
        file.write("\n\n## Automatically generated report")

    num_epochs = stats_dict["num_epochs"]

    for key, val in stats_dict.items():
        if type(val) is list:
            fig, ax = plt.subplots(figsize=(15, 15))
            sns.lineplot(
                x=range(num_epochs),
                y=stats_dict[key],
                ax=ax,
                color=colors[list(stats_dict.keys()).index(key)],
            ).set_title(key)

            fig.text(0.65, 0.85, stats_dict["hyperparams"], fontsize=10)
            fig_path = pics_path / f"{key}.png"
            fig.savefig(fig_path)
            report_text = f"\n\n**{key}** plot \n\n ![{key}]({fig_path}?raw=true)"
            with markdown_path.open("a") as file:
                file.write(report_text)
