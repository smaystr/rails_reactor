import pandas as pd
from logistic_regression import LogisticRegression
from utils import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


def split_x_y(data, y_col):
    return data.drop([y_col], axis=1).values, np.array(data[y_col]).reshape((-1, 1))


def run():
    data = pd.read_csv("data/heart_train.csv")
    test = pd.read_csv("data/heart_test.csv")

    y_col = "target"

    X, y = split_x_y(data, y_col)
    X_test, y_test = split_x_y(test, y_col)

    scaler = StandardScaler()
    scaler.fit(X)

    X = scaler.transform(X)
    X_test = scaler.transform(X_test)

    lr = LogisticRegression(lr=0.1, max_iter=15000, C=0.3, penalty="l1")
    lr.fit(X, y)

    threshold = 0.5
    print(
        f"Train score: {accuracy_score(y,lr.predict(X,threshold=threshold))}\nTest score: {accuracy_score(y_test,lr.predict(X_test,threshold=threshold))}"
    )


if __name__ == "__main__":
    run()
