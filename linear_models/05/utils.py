import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def preprocess_data(path_name, categorical_columns, target_column="target"):

    paths_for_csv = sorted(glob.glob(path_name + "*.csv"), reverse=True)

    if not len(paths_for_csv):
        raise Exception("Script should be run in a dir with the .csv files ")

    print(f"Files will be parsed {paths_for_csv}")
    train_df, test_df = [pd.read_csv(path) for path in paths_for_csv]

    full_df = pd.concat((train_df, test_df)).reset_index().drop(["index"], axis=1)

    targets = full_df[[target_column]].values
    full_df.drop([target_column], axis=1, inplace=True)

    num_columns = list(set(full_df.columns) - set(categorical_columns))

    scaler = StandardScaler()

    full_df[num_columns] = scaler.fit_transform(full_df[num_columns])

    full_df_encoded = pd.get_dummies(full_df, columns=categorical_columns)

    X_train, X_test, Y_train, Y_test = (
        full_df_encoded[: len(train_df)],
        full_df_encoded[len(train_df) :],
        targets[: len(train_df)],
        targets[len(train_df) :],
    )

    return X_train, X_test, Y_train, Y_test
