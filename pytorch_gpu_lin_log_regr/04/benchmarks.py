import argparse
from torch.optim import SGD, Adam

from utils import *
import settings
from linear_models_high_level import *
from linear_models_low_level import TorchLinearRegression, TorchLogisticRegression

config = load_config()

def test_low_level_api():

    print("Testing log_reg written on low level api")
    X_train, X_test, Y_train, Y_test = load_data_for_classification()

    log_reg = TorchLogisticRegression(
        config.get("LEARNING_RATE"), config.get("NUM_ITERATIONS"), config.get("C")
    )
    log_reg.fit(X_train, Y_train)
    print(f"Accuracy on test data is {accuracy(log_reg.predict(X_test),Y_test)}")

    print("Testing lin_reg written on low level api")
    X_train, X_test, Y_train, Y_test = load_data_for_regression()

    lin_reg = TorchLinearRegression(
        config.get("LEARNING_RATE"), config.get("NUM_ITERATIONS"), config.get("LAM")
    )
    lin_reg.fit(X_train, Y_train)
    print(f"RMSE on test data is {rmse(lin_reg.predict(X_test), Y_test)}")


def fit_model(task_name):

    print(f"Testing high level api for {task_name}")

    is_classification = task_name == "classification"

    data_loader = (
        load_data_for_classification if is_classification else load_data_for_regression
    )
    X_train, X_test, Y_train, Y_test = data_loader()
    train_loader, test_loader = create_loaders(X_train, X_test, Y_train, Y_test)

    model = LinearModel(X_train.shape[1], apply_sigmoid=is_classification)
    if config.get("USE_GPU"):
        model = model.cuda()

    optimizer_class = Adam if config.get("OPTIMIZER") == "Adam" else SGD
    optimizer = optimizer_class(
        params=model.parameters(), lr=config.get("LEARNING_RATE")
    )

    metric = accuracy if is_classification else rmse
    loss = nn.BCELoss() if is_classification else nn.MSELoss()

    training_stats = train_model(
        model, optimizer, loss, train_loader, config.get("NUM_ITERATIONS"), metric
    )

    generate_report(training_stats)

    _, metric_val = test_model(model, test_loader, metric)
    print(f"{metric.__name__} on test_data is {metric_val:.4f}")


def test_high_level_api():
    for task_type in ["classification", "regression"]:
        fit_model(task_type)


def parse_args():

    p = argparse.ArgumentParser(
        description="""
        Argument parsed for an automated model training
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config_path",
        type=pathlib.Path,
        default="./config.json",
        help="Absolute path where csv file is stored",
    )

    p.add_argument(
        "--report_path",
        type=pathlib.Path,
        default="./report.md",
        help="output path for profile report",
    )

    p.add_argument(
        "--data_path",
        type = pathlib.Path,
        default="./",
        help="path where data is located",
    )

    return p.parse_args()


def main():

    args = parse_args()

    settings.REPORT_PATH = args.report_path
    settings.CONFIG_PATH = args.config_path
    settings.DATA_PATH = args.data_path

    test_low_level_api()
    test_high_level_api()


if __name__ == "__main__":
    main()
