import argparse
import json
import torch
import seaborn as sns
import pathlib

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.nn import CrossEntropyLoss

import data
import models
import utilities
import model_validation

HEART_TRAIN_CSV = 'heart_train.csv'
HEART_TEST_CSV = 'heart_test.csv'
INSURANCE_TRAIN_CSV = 'insurance_train.csv'
INSURANCE_TEST_CSV = 'insurance_test.csv'

CONFIGURE_FILE = 'config.json'

OUTPUT_MODEL = 'output.model'
OUTPUT_INFO = 'output.info'


def get_argparse_namespace():
    parser = argparse.ArgumentParser(
        description='CPU/GPU processing'
    )
    parser.add_argument(
        '-t',
        '--problem_type',
        type=str,
        default='classification',
        choices=['classification', 'regression'],
        help='CPU or GPU'
    )
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help='CPU or GPU'
    )
    parser.add_argument(
        '-p',
        '--path',
        type=str,
        default='data',
        help='Set up the directory, where to save the worked out graphs.'
    )
    return parser.parse_args()


def generate_report(stats, where=None):
    """
    Graph report
    :type where: str
    :type stats: dict
    """
    if where is not None:
        path = pathlib.Path(where)
        if not path.exists():
            path.mkdir(parents=True)
    num_epochs = stats['num_epochs']
    for key, value in stats.items():
        if not type(value) is list:
            continue
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.lineplot(
            x=range(num_epochs),
            y=value,
            ax=ax
        ).set_title(key)
        addition = pathlib.Path(f"{key}.png")
        fig_path = path / addition
        fig.savefig(fig_path)


def main():
    args = get_argparse_namespace()

    with open(CONFIGURE_FILE, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda') \
        if (args.device == 'gpu' and torch.cuda.is_available()) \
        else torch.device('cpu')

    if args.problem_type == 'classification':
        target = 'target'
        categorial_features = ['cp', 'slope', 'ca', 'thal']
        boolean_features = ['sex', 'fbs', 'restecg', 'exang']
        X, y = utilities.load_dataset(
            train_csv=HEART_TRAIN_CSV,
            test_csv=HEART_TEST_CSV,
            target=target,
            categorial_features=categorial_features,
            boolean_features=boolean_features
        )
    else:
        target = 'charges'
        categorial_features = ['sex', 'children', 'smoker', 'region']
        X, y = utilities.load_dataset(
            train_csv=INSURANCE_TRAIN_CSV,
            test_csv=INSURANCE_TEST_CSV,
            target=target,
            categorial_features=categorial_features
        )

    train_loader, test_loader, validation_loader = data.prepare_data(
        X=X,
        y=y,
        batch_size=config['batch_size'],
        test_size=config['test_size'],
        valid_size=config['validation_size']
    )

    if args.problem_type == 'classification':
        model = models.LogisticRegression(
            input_size=X.shape[1],
            output_size=1,
            device=device
        ).double()
        criterion = torch.nn.BCELoss()
        metric = accuracy_score
    else:
        model = models.LinearRegression(
            input_size=X.shape[1],
            output_size=1,
            device=device
        ).double()
        criterion = torch.nn.MSELoss()
        metric = mean_squared_error

    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config['alpha']
        )
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            betas=(config['beta1'], config['beta2']),
            lr=config['alpha']
        )
    else:
        raise AttributeError(
            'Error! Define the optimizer correctly! (Adam/SGD)'
        )

    stats = model_validation.train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        metric=metric,
        device=device,
        print_step=config['print_step']
    )
    validation_predictions, validation_metric_value = model_validation.test_model(
        model=model,
        loader=validation_loader,
        metric=metric,
        device=device,
        isValidation=True
    )
    print(f'Metric on validation set: {validation_metric_value}')
    test_predictions, test_metric_value = model_validation.test_model(
        model=model,
        loader=test_loader,
        metric=metric,
        device=device,
        isValidation=False
    )
    print(f'Metric on test set: {test_metric_value}')
    generate_report(stats, where=args.path)


if __name__ == '__main__':
    main()
