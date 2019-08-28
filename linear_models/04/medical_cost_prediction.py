import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.regression_metrics import all_metrics
from utils.dataset_processing import normalize_data
from models.linear_regression import LinearReg


def main(train_url: str, test_url: str) -> None:
    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)

    train = pd.get_dummies(train, columns=["sex", "smoker", "region"])
    test = pd.get_dummies(test, columns=["sex", "smoker", "region"])

    X_train, y_train = (
        train.drop(columns="charges").to_numpy(),
        train["charges"].to_numpy(),
    )
    X_test, y_test = test.drop(columns="charges").to_numpy(), test["charges"].to_numpy()

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    classifier = LinearReg()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = classifier.score(X_test, y_test)

    print(f"Original values: {y_test[:5]}")

    print(f"\nMy Linear Regression")
    print(f"Predicted values: {y_pred[:5]}")

    all_metrics(y_test, y_pred)

    X_train, y_train = train.drop(columns="charges").to_numpy(), train["charges"].to_numpy()
    X_test, y_test = test.drop(columns="charges").to_numpy(), test["charges"].to_numpy()

    skl_classifier = LinearRegression()
    skl_classifier.fit(X_train, y_train)
    y_pred_skl = skl_classifier.predict(X_test)

    print(f"\nSklearn Linear Regression")
    print(f"Predicted values: {y_pred_skl[:5]}")

    all_metrics(y_test, y_pred_skl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical cost prediction")

    parser.add_argument(
        "--train_url",
        help="url/path to download train dataset",
        required=False,
        default="http://ps2.railsreactor.net/datasets/medicine/insurance_train.csv",
        type=str,
    )
    parser.add_argument(
        "--test_url",
        help="url/path to download test dataset",
        required=False,
        default="http://ps2.railsreactor.net/datasets/medicine/insurance_test.csv",
        type=str,
    )
    args = parser.parse_args()
    train_url = args.train_url
    test_url = args.test_url

    main(train_url, test_url)
