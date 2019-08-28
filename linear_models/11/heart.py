import pandas as pd
from pathlib import Path
import argparse

import linear_models


def preprocess_heart(df: pd.DataFrame):
    return df.drop('target', axis=1).values, df['target'].values


def main():

    parser = argparse.ArgumentParser(description='Logistic regression on Heart Disease UCI dataset')
    parser.add_argument('dataset_path', type=Path, help='path to dataset directory')
    args = parser.parse_args()
    
    dataset_path = args.dataset_path

    train_data = pd.read_csv(dataset_path / 'heart_train.csv')
    test_data = pd.read_csv(dataset_path / 'heart_test.csv')

    X_train, y_train = preprocess_heart(train_data)
    X_test, y_test = preprocess_heart(test_data)

    scaler = linear_models.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log_reg = linear_models.LogisticRegression()
    log_reg.fit(X_train, y_train)

    print("coefficients:", log_reg.coef_.flatten())

    print("Train accuracy", log_reg.score_)
    print("Test accuracy", log_reg.score(X_test, y_test))



if __name__ == '__main__':
    main()
