import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.classification_metrics import all_metrics, confusion_matrix
from utils.dataset_processing import normalize_data
from models.logistic_regression import LogisticReg


def main(train_url: str, test_url: str) -> None:
    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)

    train = pd.get_dummies(
        train, columns=["sex", "cp", "restecg", "slope", "ca", "thal"]
    )
    test = pd.get_dummies(
        test, columns=["sex", "cp", "restecg", "slope", "ca", "thal"]
    )

    X_train, y_train = (
        train.drop(columns="target").drop(columns=["restecg_2", "ca_4"]).to_numpy(),
        train["target"].to_numpy(),
    )
    X_test, y_test = test.drop(columns="target").to_numpy(), test["target"].to_numpy()

    normalize_columns = [0, 1, 2, 4, 6]
    X_train = normalize_data(X_train, normalize_columns)
    X_test = normalize_data(X_test, normalize_columns)

    print(f"Original values: {y_test[:5]}")

    classifier = LogisticReg(C=980)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = classifier.score(X_test, y_test)

    print(f"\nMy Logistic Regression")
    print(f"Predicted values: {y_pred[:5]}\n")

    conf_matrix = confusion_matrix(y_test, y_pred)
    all_metrics(y_test, y_pred)

    X_train, y_train = (
        train.drop(columns="target").drop(columns=["restecg_2", "ca_4"]).to_numpy(),
        train["target"].to_numpy(),
    )
    X_test, y_test = test.drop(columns="target").to_numpy(), test["target"].to_numpy()

    skl_classifier = LogisticRegression(solver="liblinear")
    skl_classifier.fit(X_train, y_train)
    y_pred_skl = skl_classifier.predict(X_test)

    print(f"\nSklearn Logistic Regression")
    print(f"Predicted values: {y_pred_skl[:5]}")

    all_metrics(y_test, y_pred_skl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart disease prediction")

    parser.add_argument(
        "--train_url",
        help="url/path to download train dataset",
        required=False,
        default="http://ps2.railsreactor.net/datasets/medicine/heart_train.csv",
        type=str,
    )
    parser.add_argument(
        "--test_url",
        help="url/path to download test dataset",
        required=False,
        default="http://ps2.railsreactor.net/datasets/medicine/heart_test.csv",
        type=str,
    )
    args = parser.parse_args()
    train_url = args.train_url
    test_url = args.test_url

    main(train_url, test_url)
