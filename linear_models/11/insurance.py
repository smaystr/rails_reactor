import pandas as pd
from pathlib import Path
import argparse
import linear_models


def preprocess_insurance(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=['region'])
    df['sex'] = df['sex'].map({'female': 1, 'male': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    return df.drop('charges', axis=1).values, df['charges'].values


def main():

    parser = argparse.ArgumentParser(description='Linear regression on insurance dataset')
    parser.add_argument('dataset_path', type=Path, help='path to dataset directory')
    args = parser.parse_args()
    
    dataset_path = args.dataset_path

    train_data = pd.read_csv(dataset_path / 'insurance_train.csv')
    test_data = pd.read_csv(dataset_path / 'insurance_test.csv')

    X_train, y_train = preprocess_insurance(train_data)
    X_test, y_test = preprocess_insurance(test_data)

    scaler = linear_models.StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lin_reg = linear_models.LinearRegression()
    lin_reg.fit(X_train, y_train)

    print("coefficients:", lin_reg.coef_.flatten())
    print("Train MSE", lin_reg.score_)
    print("Test MSE", lin_reg.score(X_test, y_test))


if __name__ == '__main__':
    main()
