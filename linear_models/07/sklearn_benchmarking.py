import logging

from sklearn.linear_model import LogisticRegression, LinearRegression

from hw3.utilities import Dataset


def test_sklearn_logreg(data: Dataset):
    lr_sk = LogisticRegression(C=0.1, max_iter=100000, solver="liblinear")
    lr_sk.fit(data.X_train, data.Y_train)
    logging.info(f'SK model score is: {lr_sk.score(data.X_test, data.Y_test)}')


def test_sklearn_linreg(data: Dataset):
    lr_sk = LinearRegression()
    lr_sk.fit(data.X_train, data.Y_train)
    logging.info(f'SK model score is: {lr_sk.score(data.X_test, data.Y_test)}')
