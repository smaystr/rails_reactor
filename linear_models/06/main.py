import argparse
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from load_data import load_data, normalize
from metrics import *

def main(model: str, train: str, test: str):
    X, y = load_data(train, model)
    X_test, y_test = load_data(test, model)
    X = normalize(X)
    X_test = normalize(X_test)
    
    if model == 'logistic':
        lr = LogisticRegression().fit(X, y)
        y_pred_train = lr.predict(X)
        y_pred_test = lr.predict(X_test)

        print('Coefficients:\n', lr.coef)
        print('Score fot training set:\n', accuracy(y, y_pred_train))
        print('Score for test set:\n', accuracy(y_test, y_pred_test))
        print('Prediction for test set (first 5 rows):\n', lr.predict(X_test)[0:5, :])
        print('Probability prediction for test set (first 5 rows):\n', lr.predict_proba(X_test)[0:5, :])
    elif model == 'linear':
        lr = LinearRegression().fit(X, y)
        y_pred_train = lr.predict(X)
        y_pred_test = lr.predict(X_test)

        print('Coefficients:\n', lr.coef)
        print('Score fot training set:\n', mse(y, y_pred_train))
        print('Score for test set:\n', mse(y_test, y_pred_test))
        print('Prediction for test set (first 5 rows):\n', lr.predict(X_test)[0:5, :])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='type of regression (linear or logistic)', type=str, required=True)
    parser.add_argument('--train', help='path to the train dataset', type=str, required=True)
    parser.add_argument('--test', help='path to the test dataset', type=str, required=True)
    args = parser.parse_args()

    main(args.model, args.train, args.test)